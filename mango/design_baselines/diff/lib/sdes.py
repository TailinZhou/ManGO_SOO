import torch
from lib.utils import sample_v, log_normal, sample_vp_truncated_q
import numpy as np
import torch.nn.functional as F


class VariancePreservingSDE(torch.nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max-self.beta_min)*t

    def mean_weight(self, t):
        return torch.exp(-0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min)

    def var(self, t):
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min)

    def f(self, t, y):
        return - 0.5 * self.beta(t) * y

    def g(self, t, y):
        beta_t = self.beta(t)
        return torch.ones_like(y) * beta_t**0.5

    def sample(self, t, y0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        mu = self.mean_weight(t) * y0
        std = self.var(t) ** 0.5
        epsilon = torch.randn_like(y0)
        yt = epsilon * std + mu
        if not return_noise:
            return yt
        else:
            return yt, epsilon, std, self.g(t, yt)

    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)


class ScorePluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """

    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias

    # Drift
    def mu(self, t, y, ya, lmbd=0., gamma=0.):
        a = self.a(y, self.T - t.squeeze(), ya) * (1 + gamma) - gamma * self.a(y, self.T - t.squeeze(), torch.zeros_like(ya))
        return (1. - 0.5 * lmbd) * (self.base_sde.g(self.T-t, y) ** 2) *  a - \
               self.base_sde.f(self.T - t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    @torch.enable_grad()
    def dsm(self, x, y):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        a = self.a(x_hat, t_.squeeze(), y)

        return ((a * std + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def dsm_weighted(self, x, y, w, clip=False, c_min=None, c_max=None):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)

        if clip:
            c_min = c_min.repeat((x.size(0),1))
            c_max = c_max.repeat((x.size(0),1))
            c_min = c_min.cuda()
            c_max = c_max.cuda()
            x_hat = x_hat.cuda()

            x_hat = torch.clip(x_hat, min=c_min, max=c_max)

        a = self.a(x_hat, t_.squeeze(), y)

        return (w * ((a * std + target) ** 2)).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def elbo_random_t_slice(self, x, y_n):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.base_sde.g(t_, y) * self.a(y, t_.squeeze(), y_n)
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        v = sample_v(x.shape, vtype=self.vtype).to(y)

        Mu = - (
              torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = - (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu

class PluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """

    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias

    # Drift
    def mu(self, t, y, ya, lmbd=0., gamma=0.):
        a = self.a(y, self.T - t.squeeze(), ya) * (1 + gamma) - gamma * self.a(y, self.T - t.squeeze(), torch.zeros_like(ya))
        return (1. - 0.5 * lmbd) * self.base_sde.g(self.T-t, y) * a - \
               self.base_sde.f(self.T - t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    @torch.enable_grad()
    def dsm(self, x, y):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        a = self.a(x_hat, t_.squeeze(), y)

        return ((a * std / g + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def dsm_weighted(self, x, y, w, clip=False, c_min=None, c_max=None):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)

        if clip:
            c_min = c_min.repeat((x.size(0),1))
            c_max = c_max.repeat((x.size(0),1))
            c_min = c_min.cuda()
            c_max = c_max.cuda()
            x_hat = x_hat.cuda()

            x_hat = torch.clip(x_hat, min=c_min, max=c_max)

        a = self.a(x_hat, t_.squeeze(), y)

        return (w * ((a * std / g + target) ** 2)).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def elbo_random_t_slice(self, x, y_n):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.a(y, t_.squeeze(), y_n)
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        v = sample_v(x.shape, vtype=self.vtype).to(y)

        Mu = - (
              torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = - (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu


class UnconditionPluginReverseSDE(torch.nn.Module):
    """
    inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    f <- g a - f
    g <- g
    (time is inverted)
    """

    def __init__(self, base_sde, drift_a, T, vtype='rademacher', debias=False,forwardmodel=None, encoder=None, decoder=None):
        super().__init__()
        self.base_sde = base_sde
        self.a = drift_a
        self.T = T
        self.vtype = vtype
        self.debias = debias
        self.forwardmodel = forwardmodel
        self.encoder = encoder
        self.decoder = decoder

 

    def mu(self, t, y, ya, lmbd=0., gamma=0., guidance_bool=False, guidance_scals1=0, guidance_scals2=1, return_guidance=False, x_min_constraint=None, x_max_constraint=None):

        std_t = self.base_sde.var(t)**0.5
        mean_t = self.base_sde.mean_weight(t)  
        with torch.no_grad():
            self.a.eval()
            score =  self.a(y, t.squeeze(), ya) 
        
        if guidance_bool:
            input_size = y.size(-1) - ya.size(-1)
            with torch.enable_grad():
                y = y.requires_grad_(True)
                # Predict x0  
                y0 = (y + std_t**2 * score) / mean_t
                # Calculate the 
                loss = F.mse_loss(y0[:, input_size:], ya, reduction='none').mean(dim=-1)

                if x_min_constraint is not None or x_max_constraint is not None:
                    y0_part = y0[:, :input_size]
                    lower_violation = torch.clamp(x_min_constraint - y0_part, min=0) if x_min_constraint is not None else 0
                    upper_violation = torch.clamp(y0_part - x_max_constraint, min=0) if x_max_constraint is not None else 0
                    violation = lower_violation + upper_violation
                    loss_x_constraint = (violation ** 2).mean(dim=-1)
                    loss = loss + loss_x_constraint 

                loss.sum().backward()
                guidance = y.grad.clone().detach()

                guidance_scals1 = torch.ones_like(y[:, :input_size]) * guidance_scals1
                guidance_scals2 = torch.ones_like(y[:, input_size:]) * guidance_scals2
                guidance_scals = torch.cat((guidance_scals1, guidance_scals2), dim=-1)

                scale_vector = torch.ones_like(y)   
                scale_vector[:, input_size:] = torch.where(
                    torch.abs(guidance[:, input_size:]) < 1e-8,  # Conditions
                    1.0, # If the condition is true, set it to 1 (scalar, which will be broadcast automatically)
                    torch.abs(score[:, input_size:]) / torch.abs(guidance[:, input_size:])  # If the condition is false, calculate according to the formula
                )
                guidance = guidance_scals* scale_vector * guidance   

        else:
            guidance = torch.zeros_like(score)
        
        if return_guidance:
            return guidance
        else:
            drift = self.base_sde.beta(t) * (score - guidance) - self.base_sde.f(t, y)
            return drift
 
 


    def sigma(self, t, y, lmbd=0.):
        return   self.base_sde.g( t , y)

    @torch.enable_grad()
    def dsm(self, x, y):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)
        a = self.a(x_hat, t_.squeeze(), y)

        return ((a * std / g + target) ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def dsm_weighted(self, x, y, w, clip=False, c_min=None, c_max=None):
        """
        denoising score matching loss
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        else:
            t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)

        if clip:
            c_min = c_min.repeat((x.size(0),1))
            c_max = c_max.repeat((x.size(0),1))
            c_min = c_min.cuda()
            c_max = c_max.cuda()
            x_hat = x_hat.cuda()

            x_hat = torch.clip(x_hat, min=c_min, max=c_max)

        a = self.a(x_hat, t_.squeeze(), y)
        w_mean = torch.mean(w, dim=1, keepdim=True)
        
        return (w_mean * ((a * std / g + target) ** 2)).view(x.size(0), -1).sum(1, keepdim=False) / 2

    @torch.enable_grad()
    def elbo_random_t_slice(self, x, y_n):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        qt = 1 / self.T
        y = self.base_sde.sample(t_, x).requires_grad_()

        a = self.a(y, t_.squeeze(), y_n)
        mu = self.base_sde.g(t_, y) * a - self.base_sde.f(t_, y)

        v = sample_v(x.shape, vtype=self.vtype).to(y)

        Mu = - (
              torch.autograd.grad(mu, y, v, create_graph=self.training)[0] * v
        ).view(x.size(0), -1).sum(1, keepdim=False) / qt

        Nu = - (a ** 2).view(x.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.base_sde.sample(torch.ones_like(t_) * self.base_sde.T, x)
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x.size(0), -1).sum(1)

        return lp + Mu + Nu
