import torch
from torch import nn
from torch.functional import F
import numpy as np
from torchxs.core.math import SphericalBessel
from torchxs.core.shapes import gaussian_mixture_of_spheres, sphere
import einops

import matplotlib.pyplot as plt


class Kramer_fit(nn.Module):
    def __init__(self, q_tensor, i_obs, s_obs, radius, padding=5, porod_factor=6.0):

        super().__init__()
        self.porod_factor=porod_factor
        self.q_tensor = q_tensor

        self.radius = radius
        self.padding = padding

        self.i_obs = i_obs
        self.s_obs = s_obs

        self.max_k = int(torch.max(self.q_tensor * self.radius * 2.0 / np.pi) + 1) + self.padding
        self.alphas = torch.Tensor(np.arange(1,self.max_k+1)*np.pi)
        self.qks = self.alphas / ( 2.0 * self.radius)

        self.Skq = self.intensity_basis_functions()
        weights = torch.log(self.initial_guess())
        self.weights = nn.Parameter(weights)

    def intensity_basis_functions(self):
        t = self.q_tensor * self.radius * 2.0
        tmp_top = torch.outer(self.alphas/SphericalBessel.j1(self.alphas), SphericalBessel.j0(t) )
        tmp_bottom = torch.outer( self.alphas**2.0, torch.ones(t.shape) )
        tmp_bottom = tmp_bottom - t*t
        result = 2.0*tmp_top/tmp_bottom
        return result.T

    def initial_guess(self):
        Iqks = []
        for kk, qk in enumerate(self.qks):
            if qk < torch.max(self.q_tensor):
                this_indx = torch.argmin(torch.abs(self.q_tensor - qk))
                Iqks.append(self.i_obs[this_indx])
            else:
                ratio = ((kk+1)/(kk+2))**self.porod_factor
                Iqks.append(Iqks[-1]*ratio)
        Iqks = torch.Tensor(Iqks)
        return Iqks

    def forward(self):
        result = torch.matmul( self.Skq,torch.exp(self.weights))
        return result

class Kramer_interpolator(nn.Module):
    def __init__(self, radius, q_max=None, padding=None, Iqks=None):
        super().__init__()
        self.radius = radius
        self.q_max = q_max
        self.padding = padding
        if q_max is None:
            assert Iqks is not None
        if self.padding is None:
            assert Iqks is not None

        if Iqks is not None:
            self.max_k = Iqks.shape[-1]
        else:
            self.max_k = int(self.q_max * self.radius * 2.0 / np.pi + 1) + self.padding

        self.alphas = torch.Tensor(np.arange(1, self.max_k + 1) * np.pi)
        self.qks = self.alphas / (2.0 * self.radius)

        if Iqks is not None:
            weights = torch.log(Iqks)
        else:
            weights = torch.log(self.qks**-4.0)
        self.weights = nn.Parameter(weights)

    def intensity_basis_functions(self, q_tensor):
        t = q_tensor * self.radius * 2.0
        tmp_top = torch.outer(self.alphas / SphericalBessel.j1(self.alphas), SphericalBessel.j0(t))
        tmp_bottom = torch.outer(self.alphas ** 2.0, torch.ones(t.shape))
        tmp_bottom = tmp_bottom - t * t
        result = 2.0 * tmp_top / tmp_bottom
        return result.T

    def forward(self, q_tensor):
        Skq = self.intensity_basis_functions(q_tensor)
        result = torch.matmul(Skq, torch.exp(self.weights))
        return result





def training_loop(model, optimizer, n=1000):
    "Training loop for torch model."
    losses = []

    y = model.i_obs
    s = model.s_obs

    for i in range(n):
        preds = model()
        loss = F.l1_loss(preds/s, y/s).sqrt()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses



if __name__ == "__main__":
    q_obs_max = 0.65
    N_obs = 1350
    q_obs = torch.Tensor(np.linspace(0.005, q_obs_max, N_obs))
    Iobs = 0
    radius = 100
    ws = 0.0
    sr = .3
    Iobs = gaussian_mixture_of_spheres.i_of_q(q_obs,radius,8)+1e-8

    Iobs1 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 0.5))
    Iobs2 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 0.5))
    Iobs3 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 0.5))
    Iobs4 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 0.5))
    Iobs5 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 0.5))
    Iobs = einops.rearrange([Iobs1, Iobs2, Iobs3, Iobs4, Iobs5], "N X -> N X")

    m = torch.mean(Iobs, dim=0)
    s = torch.std(Iobs, dim=0) / 2
    plt.plot(q_obs.numpy(), m, '.')
    plt.xlim(0, q_obs_max * 1.05)
    plt.yscale('log')
    plt.show()

    obj1 = Kramer_fit(q_obs,m,s, radius+50, padding=1000)
    optim1 = torch.optim.Adam(obj1.parameters(), lr=5e-4)
    losses1 = training_loop(obj1, optim1)
    Iqks = torch.exp(obj1.weights)

    obj2 = Kramer_interpolator(radius=radius+50, Iqks=Iqks)
    q = torch.Tensor(np.linspace(0,0.65,1000))
    tmp = obj2(q).detach()

    plt.plot(q_obs,m, '.')
    yyy1 = obj1().detach().numpy().flatten()
    plt.plot(q_obs, yyy1, '--', c='r')
    plt.plot(q, tmp, '-', c='g')

    plt.yscale('log')
    plt.show()








