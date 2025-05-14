#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


class Diffusion(nn.Module):

    def __init__(
            self,
            loss_type,
            num_diffusion_timesteps = 1000,
            beta_schedule = 'sigmoid',
            beta_start = 1.0e-7,
            beta_end = 2.0e-3,
            cos_beta_s = 0.01,
        ) -> None:
        super().__init__()

        self.loss_type = loss_type
        assert loss_type in ['C0', 'noise']

        # scheduler
        if beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(num_diffusion_timesteps, cos_beta_s) ** 2
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            alphas = 1. - betas

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # for recording training
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

    def _predict_x0_from_eps(self, xt, eps, t, batch_ids):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch_ids) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch_ids) * eps
        return pos0_from_e
    
    def q_posterior(self, x0, xt, t, batch_ids):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_mean = extract(self.posterior_mean_c0_coef, t, batch_ids) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch_ids) * xt
        return pos_mean
    
    def kl_prior(self, x0, batch_ids):
        batch_size = batch_ids.max() + 1
        a = extract(self.alphas_cumprod, [self.num_timesteps - 1] * batch_size, batch_ids)
        mu = a.sqrt() * x0
        pos_log_variance = torch.log((1.0 - a).sqrt())
        kl_prior = normal_kl(torch.zeros_like(mu), torch.zeros_like(pos_log_variance),
                             mu, pos_log_variance)
        kl_prior = scatter_mean(kl_prior, batch_ids, dim=0)
        return kl_prior
    
    def sample_time(self, batch_size, device, method = 'symmetric'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(batch_size, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(batch_size // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:batch_size]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError
        
    def compute_Lt(self, mu_model, x0, xt, t, batch_ids):
        # fixed variance
        log_variance = extract(self.posterior_logvar, t, batch_ids)
        mu_true = self.q_posterior(x0=x0, xt=xt, t=t, batch_ids=batch_ids)
        kl_pos = normal_kl(mu_true, log_variance, mu_model, log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=mu_model, log_scales=0.5 * log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch_ids]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch_ids, dim=0)
        return loss_pos
    
    def perturb(self, x0, t, batch_ids):
        a = self.alphas_cumprod[t][batch_ids].unsqueeze(1) # [N, 1]
        noise = torch.zeros_like(x0)
        noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        xt = a.sqrt() * x0 + (1.0 - a).sqrt() * noise
        return xt, noise

    def get_loss(self, x_pred, xt, x0, noise_true, batch_ids, reduction='mean'):
        if self.loss_type == 'noise':
            noise_pred = x_pred - xt
            # x0_from_eps = self._predict_x0_from_eps(
            #     xt=xt, eps=noise_pred, t=t, batch_ids=batch_ids
            # )
            # mu_pred = self.q_posterior(
            #     x0=x0_from_eps, xt=xt, t=t, batch_ids=batch_ids
            # )
            loss = ((noise_pred - noise_true) ** 2).sum(-1)
        elif self.loss_type == 'C0':
            # mu = self.q_posterior(
            #     x0=x_pred, xt=xt, t=t, batch_ids=batch_ids
            # )
            loss = ((x_pred - x0) ** 2).sum(-1)
        else:
            raise ValueError

        if reduction == 'mean':
            loss = scatter_mean(loss, batch_ids, dim=0) # [batch_size]
            loss = torch.mean(loss)

        return loss
    
    @torch.no_grad()
    def sample(self, xt, batch_ids, denoise_func, num_steps=None):

        if num_steps is None: num_steps = self.num_timesteps
        batch_size = batch_ids.max() + 1

        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))

        for i in time_seq:
            t = torch.full(size=(batch_size,), fill_value=i, dtype=torch.long, device=batch_ids.device)
            x_pred = denoise_func(xt, t)

            if self.loss_type == 'noise':
                noise_pred = x_pred - xt
                x0_pred = self._predict_x0_from_eps(
                    xt=xt, eps=noise_pred, t=t, batch_ids=batch_ids
                )
            elif self.loss_type == 'C0':
                x0_pred = x_pred

            mu_pred = self.q_posterior(x0=x0_pred, xt=xt, t=t, batch_ids=batch_ids)
            log_variance = extract(self.posterior_logvar, t, batch_ids)

            # no noise when t == 0
            nonzero_mask = (1 - (t==0).float())[batch_ids].unsqueeze(-1)
            x_next = mu_pred + nonzero_mask * (0.5 * log_variance).exp() * torch.randn_like(xt)
            xt = x_next
        
        return xt