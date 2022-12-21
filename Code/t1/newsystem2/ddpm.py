import torch
from torch import nn, Tensor
from torch.nn import Identity, Module, ModuleList, Linear, Sigmoid, GroupNorm, Conv2d, ConvTranspose2d, Dropout
from typing import Union, Tuple, List, Optional, Callable
import enum
from tqdm.auto import tqdm
from utils import normal_kl, mean_flat, discretized_gaussian_log_likelihood, gather, extract
import numpy as np
import torchvision as tv
import torchvision.transforms.functional as fn
from scipy import linalg

class DDPM:
    def __init__(self, model: Module, T: int, device: torch.device):
        self.model = model
        self.T = T
        self.device = device
        
        # linear beta_t schedule
        self.beta = torch.linspace(0.0001, 0.002, self.T, dtype=torch.float64)
        # TODO: other beta schedules
        # TODO: other t values

        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = torch.cat([Tensor([1.0]), self.alpha_bar[:-1]])
        self.alpha_bar_next = torch.cat([self.alpha_bar[1:], Tensor([0.0])])

        # calcs for q(x_t | x_t-1)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_1m_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.log_1m_alpha_bar = torch.log(1.0 - self.alpha_bar)
        self.sqrt_recipr_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recipr_alpha_bar_m1 =  torch.sqrt((1.0 / self.alpha_bar) - 1.0)

        # calcs for q(x_t-1 | x_t, x_0) (posterior)
        self.post_var = (self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar))
        self.log_clip_post_var = torch.log(torch.cat([Tensor([self.post_var[1]]), self.post_var[1:]]))
        self.post_mean_coef_1 = (self.beta * torch.sqrt(self.alpha_bar_prev)) / (1.0 - self.alpha_bar)
        self.post_mean_coef_2 = (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alpha) / (1.0 - self.alpha_bar)

    def q_xt_x0(self, x0: Tensor, t: Tensor):
        """get q(x_t | x_0) distribution."""
        mean = extract(self.sqrt_alpha_bar, t, x0.shape) * x0
        var = extract(1.0 - self.alpha_bar, t, x0.shape)
        log_var = extract(self.log_1m_alpha_bar, t, x0.shape)
        return mean, var, log_var

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None):
        """sample from q(x_t | x_0)."""
        noise = torch.randn_like(x0) if noise is None else noise
        # get distribution
        # mean, var, _ = self.q_xt_x0(x0, t)
        mean = extract(self.sqrt_alpha_bar, t, x0.shape) * x0
        var = extract(self.sqrt_1m_alpha_bar, t, x0.shape)
        # sample
        return mean + var * noise

    def q_xtm1_xt_x0(self, x0: Tensor, xt: Tensor, t: Tensor):
        """get q(x_t-1 | x_t, x_0) distribution. (posterior)"""
        mean = (extract(self.post_mean_coef_1, t, xt.shape) * x0) + (extract(self.post_mean_coef_2, t, xt.shape) * xt)
        var = extract(self.post_var, t, xt.shape)
        log_clip_var = extract(self.log_clip_post_var, t, xt.shape)
        return mean, var, log_clip_var


    def p_xtm1_xt(self, xt: Tensor, t: Tensor, clip_denoised: bool = True, denoise_fn: Callable = None):
        model_output = self.model(xt, t)

        model_variance = torch.cat([Tensor([self.post_var[1]]), self.beta[1:]])
        model_variance = extract(model_variance, t, xt.shape)
        model_log_variance = torch.log(torch.cat([Tensor([self.post_var[1]]), self.beta[1:]]))
        model_log_variance = extract(model_log_variance, t, xt.shape)

        pred_x0 = (extract(self.sqrt_recipr_alpha_bar, t, xt.shape) * xt) - (extract(self.sqrt_recipr_alpha_bar_m1, t, xt.shape) * model_output)

        if denoise_fn is not None:
            pred_x0 = denoise_fn(xt)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)

        model_mean, _, _ = self.q_xtm1_xt_x0(x0=pred_x0, xt=xt, t=t)

        return model_mean, model_variance, model_log_variance, pred_x0


    def p_sample(self, xt: Tensor, t: Tensor, clip_denoised: bool = True, denoise_fn: Callable = None):
        """sample from p_theta(x_t-1 | x_t)"""
        noise = torch.randn_like(xt)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(xt.shape) - 1))))  # no noise when t == 0
        # get distribution
        mean, var, log_var, pred_x0 = self.p_xtm1_xt(xt, t, clip_denoised, denoise_fn)
        # sample
        sample = mean + ((nonzero_mask * torch.exp(0.5 * log_var)) * noise)
        return sample, pred_x0

    def p_sample_loop(self, shape, noise=None, clip_denoised=True, denoise_fn=None):
        samples = []
        for s in self.p_sample_loop_prog(shape, noise=noise, clip_denoised=clip_denoised, denoise_fn=denoise_fn):
            img = s[0]
            # img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            # img = img.permute(0, 2, 3, 1)
            # img = img.contiguous()
            samples.append(img)
        return samples

    def p_sample_loop_prog(self, shape, noise=None, clip_denoised=True, denoise_fn=None):
        img = noise if noise is not None else torch.randn(*shape, device=self.device)
        indices = tqdm(reversed(range(self.T)))
        for i in indices:
            t = torch.tensor([i] * shape[0], device=self.device)
            with torch.no_grad():
                sample, pred_x0 = self.p_sample(img, t, clip_denoised=clip_denoised, denoise_fn=denoise_fn)
                yield (sample, pred_x0)
                img = sample

    def loss(self, x0: Tensor, t, noise: Optional[Tensor] = None):
        noise = torch.randn_like(x0) if noise is None else noise
        xt = self.q_sample(x0, t, noise)
        model_output = self.model(xt, t)
        se = (noise - model_output) ** 2
        mse = mean_flat(se)
        mse = torch.mean(mse)
        return mse

    def vlb_calc(self, x0: Tensor, t, xt: Tensor = None, clip_denoised=True, noise = None):
        noise = torch.randn_like(x0) if noise is None else noise
        xt = self.q_sample(x0, t, noise) if xt == None else xt
        q_mean, q_var, q_log_clip_var = self.q_xtm1_xt_x0(x0, xt, t)
        p_mean, p_var, p_log_var, p_pred_x0  = self.p_xtm1_xt(xt, t, clip_denoised=clip_denoised)
        kl = normal_kl(q_mean, q_log_clip_var, p_mean, p_log_var)
        kl = mean_flat(kl) / torch.log(Tensor([2.0]).to(self.device))
        decoder_nll = -discretized_gaussian_log_likelihood(x=x0, means=p_mean, log_scales=0.5*p_log_var)
        decoder_nll = mean_flat(decoder_nll) / torch.log(Tensor([2.0]).to(self.device))
        vlb = torch.where((t == 0), decoder_nll, kl)
        return vlb, p_pred_x0

    def bpd_loop(self, x0, clip_denoised=True):
        batch_size = x0.shape[0]
        vlbs = []
        x0_mses = []
        mses = []
        for t in tqdm(reversed(range(self.T))):
            t_batch = torch.tensor([t] * batch_size, device=self.device)
            noise = torch.randn_like(x0)
            xt = self.q_sample(x0=x0, t=t_batch, noise=noise)
            with torch.no_grad():
                vlb, pred_x0 = self.vlb_calc(
                    x0=x0,
                    t=t_batch,
                    xt=xt,
                    clip_denoised=clip_denoised,
                )
            vlbs.append(vlb)
            x0_mses.append(mean_flat(pred_x0 - x0) ** 2)
            eps = (extract(self.sqrt_recipr_alpha_bar, t_batch, xt.shape) * xt - pred_x0) / extract(self.sqrt_recipr_alpha_bar_m1, t_batch, xt.shape)
            mse = mean_flat((eps - noise) ** 2)
            mses.append(mse)

        vlbs = torch.stack(vlbs, dim=1)
        x0_mses = torch.stack(x0_mses, dim=1)
        mses = torch.stack(mses, dim=1)

        prior_bpd = self.prior_bpd(x0)
        total_bpd = vlbs.sum(dim=1) + prior_bpd

        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vlbs": vlbs,
            "x0_mses": x0_mses,
            "mses": mses,
        }

    def prior_bpd(self, x0):
        batch_size = x0.shape[0]
        t = torch.tensor([self.T - 1] * batch_size, device=x0.device)
        qt_mean, _, qt_log_variance = self.q_xt_x0(x0, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    # FROM pytorch-fid
    def calculate_activation_statistics(self, images, dims=2048):
        images = fn.resize(images, (299, 299))
        self.inception = tv.models.inception_v3(weights='DEFAULT', transform_input=False).to(self.device)
        self.inception.eval()
        act = np.empty((len(images), dims))

        batch = images.to(self.device)
        pred = self.inception(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        act = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    # FROM pytorch-fid
    def calculate_frechet_distance(self, mu_1, sigma1, mu_2, sigma2, eps=1e-6):
        mu_1 = np.atleast_1d(mu_1)
        mu_2 = np.atleast_1d(mu_2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu_1.shape == mu_2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu_1 - mu_2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def calculate_frechet(self, images_real, images_fake):
        mu_1, std_1 = self.calculate_activation_statistics(images_real)
        mu_2, std_2 = self.calculate_activation_statistics(images_fake)

        fid_value = self.calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
        return fid_value


# class DDPM_OLD:
#     def __init__(self, model: Module, T: int, device: torch.device, beta_schedule: str):
#         self.model = model
#         self.T = T
        
#         # linear beta_t schedule
#         self.beta = torch.linspace(0.0001, 0.002, self.T, dtype=torch.float64)
#         # TODO: other beta schedules
#         # TODO: other t values

#         self.alpha = 1. - self.beta
#         self.alpha_bar = torch.cumprod(self.alpha, dim=0)
#         self.sigma_squared = self.beta

#     def q_xt_x0(self, x0: Tensor, t: Tensor):
#         """get q(x_t | x_0) distribution."""
#         g = gather(self.alpha_bar, t)
#         mean = g ** 0.5 * x0
#         var = 1 - g
#         return mean, var

#     def p_xtm1_xt(self, xt: Tensor, t: Tensor):
#         epsilon_theta = self.model(xt, t)
#         alpha = gather(self.alpha, t)
#         alpha_bar = gather(self.alpha_bar, t)
#         coef = (1 - alpha) / ((1 - alpha_bar) ** 0.5)
#         mean = 1 / ((alpha ** 0.5) * (xt - (coef * epsilon_theta)))
#         var = gather(self.sigma_squared, t)
#         return mean, var

#     def q_sample(self, x0: Tensor, t: Tensor, epsilon: Optional[Tensor] = None):
#         """sample from q(x_t | x_0)."""
#         epsilon = torch.rand_like(x0) if epsilon is None else epsilon
#         # get distribution
#         mean, var = self.q_xt_x0(x0, t)
#         # sample
#         return mean + (var ** 0.5) * epsilon

#     def p_sample(self, xt: Tensor, t: Tensor):
#         """sample from p_theta(x_t-1 | x_t)"""
#         epsilon = torch.randn(xt.shape, device=xt.device)
#         # get distribution
#         mean, var = self.p_xtm1_xt(xt, t)
#         # sample
#         return mean + (var ** 0.5) * epsilon 

#     def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
#         batch_size = x0.shape[0]
#         # Get random $t$ for each sample in the batch
#         t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

#         # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
#         if noise is None:
#             noise = torch.randn_like(x0)

#         # Sample $x_t$ for $q(x_t|x_0)$
#         xt = self.q_sample(x0, t, eps=noise)
#         # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
#         eps_theta = self.model(xt, t)

#         # MSE loss
#         return F.mse_loss(noise, eps_theta)