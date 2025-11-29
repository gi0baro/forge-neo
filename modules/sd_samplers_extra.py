import k_diffusion.sampling
import torch
import tqdm

from modules.uni_pc import uni_pc as unipc_impl


@torch.no_grad()
def restart_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, restart_list=None):
    """
    Implements restart sampling in Restart Sampling for Improving Generative Processes (2023)
    Restart_list format: {min_sigma: [ restart_steps, restart_times, max_sigma]}
    If restart_list is None: will choose restart_list automatically, otherwise will use the given restart_list
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0
    from k_diffusion.sampling import get_sigmas_karras, to_d

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        nonlocal step_id
        denoised = model(x, old_sigma * s_in, **extra_args)
        d = to_d(x, old_sigma, denoised)
        if callback is not None:
            callback({"x": x, "i": step_id, "sigma": new_sigma, "sigma_hat": old_sigma, "denoised": denoised})
        dt = new_sigma - old_sigma
        if new_sigma == 0 or not second_order:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        step_id += 1
        return x

    steps = sigmas.shape[0] - 1
    if restart_list is None:
        if steps >= 20:
            restart_steps = 9
            restart_times = 1
            if steps >= 36:
                restart_steps = steps // 4
                restart_times = 2
            sigmas = get_sigmas_karras(steps - restart_steps * restart_times, sigmas[-2].item(), sigmas[0].item(), device=sigmas.device)
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            restart_list = {}

    restart_list = {int(torch.argmin(abs(sigmas - key), dim=0)): value for key, value in restart_list.items()}

    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_list:
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(restart_steps, sigmas[min_idx].item(), sigmas[max_idx].item(), device=sigmas.device)[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_list.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    last_sigma = None
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        if last_sigma is None:
            last_sigma = old_sigma
        elif last_sigma < old_sigma:
            x = x + k_diffusion.sampling.torch.randn_like(x) * s_noise * (old_sigma**2 - last_sigma**2) ** 0.5
        x = heun_step(x, old_sigma, new_sigma)
        last_sigma = new_sigma

    return x


class SigmaConvert:
    schedule = ""

    def marginal_log_mean_coeff(self, sigma):
        return 0.5 * torch.log(1 / ((sigma * sigma) + 1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T]
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std


def predict_eps_sigma(model, input, sigma_in, **kwargs):
    sigma = sigma_in.view(sigma_in.shape[:1] + (1,) * (input.ndim - 1))
    input = input * ((sigma**2 + 1.0) ** 0.5)
    return (input - model(input, sigma_in, **kwargs)) / sigma


def sample_unipc(model, x, sigmas, extra_args=None, callback=None, disable=False, variant="bh1"):
    timesteps = sigmas.clone()
    if sigmas[-1] < 0.001:
        timesteps[-1] = 0.001

    ns = SigmaConvert()

    x = x / torch.sqrt(1.0 + timesteps[0] ** 2.0)
    model_type = "noise"

    model_fn = unipc_impl.model_wrapper(
        lambda input, sigma, **kwargs: predict_eps_sigma(model, input, sigma, **kwargs),
        ns,
        model_type=model_type,
        guidance_type="uncond",
        model_kwargs=extra_args,
    )

    order = min(3, len(timesteps) - 2)
    uni_pc = unipc_impl.UniPC(model_fn, ns, predict_x0=True, thresholding=False, variant=variant)
    x = uni_pc.sample(x, timesteps=timesteps, skip_type="time_uniform", method="multistep", order=order, lower_order_final=True, callback=callback, disable_pbar=disable)
    x /= ns.marginal_alpha(timesteps[-1])
    return x


@torch.no_grad()
def er_sde_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None, noise_scaler=None, max_stage=3):
    """
    Extended Reverse-Time SDE solver (VE ER-SDE-Solver-3). Arxiv: https://arxiv.org/abs/2309.06169.
    Code reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py.
    """
    from k_diffusion.sampling import default_noise_sampler
    from tqdm.auto import trange

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    def default_noise_scaler(sigma):
        return sigma * ((sigma ** 0.3).exp() + 10.0)

    noise_scaler = default_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        elif stage_used == 1:
            r = noise_scaler(sigmas[i + 1]) / noise_scaler(sigmas[i])
            x = r * x + (1 - r) * denoised
        else:
            r = noise_scaler(sigmas[i + 1]) / noise_scaler(sigmas[i])
            x = r * x + (1 - r) * denoised

            dt = sigmas[i + 1] - sigmas[i]
            sigma_step_size = -dt / num_integration_points
            sigma_pos = sigmas[i + 1] + point_indice * sigma_step_size
            scaled_pos = noise_scaler(sigma_pos)

            # Stage 2
            s = torch.sum(1 / scaled_pos) * sigma_step_size
            denoised_d = (denoised - old_denoised) / (sigmas[i] - sigmas[i - 1])
            x = x + (dt + s * noise_scaler(sigmas[i + 1])) * denoised_d

            if stage_used >= 3:
                # Stage 3
                s_u = torch.sum((sigma_pos - sigmas[i]) / scaled_pos) * sigma_step_size
                denoised_u = (denoised_d - old_denoised_d) / ((sigmas[i] - sigmas[i - 2]) / 2)
                x = x + ((dt ** 2) / 2 + s_u * noise_scaler(sigmas[i + 1])) * denoised_u
            old_denoised_d = denoised_d

        if s_noise != 0 and sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (sigmas[i + 1] ** 2 - sigmas[i] ** 2 * r ** 2).sqrt()
        old_denoised = denoised
    return x
