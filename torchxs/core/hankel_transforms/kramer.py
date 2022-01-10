import torch
import torch.optim
from torch import nn
from torch.functional import F
import numpy as np
from torchxs.core.math import SphericalBessel
from torchxs.core.shapes import gaussian_mixture_of_spheres, sphere
import einops

import matplotlib.pyplot as plt

class Kramer_fit(nn.Module):
    def __init__(self, q_tensor, i_obs, s_obs, radius, padding=5, porod_factor=8.0):
        super().__init__()

        self.max_w = 140.0
        self.min_w = -180

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
        weights = self.initial_guess()
        self.weights = nn.Parameter(torch.log(torch.abs(weights)))
        #self.log_scale = nn.Parameter(torch.Tensor([0]))
        self.log_bg = nn.Parameter( torch.min(torch.log(torch.abs(weights)/20.0)) )

    def intensity_basis_functions(self):
        t = self.q_tensor * self.radius * 2.0
        tmp_top = torch.outer(self.alphas/SphericalBessel.j1(self.alphas), SphericalBessel.j0(t) )
        tmp_bottom = torch.outer( self.alphas**2.0, torch.ones(t.shape) )
        tmp_bottom = tmp_bottom - t*t
        result = 2.0*tmp_top/tmp_bottom
        return result.T

    def initial_guess(self, delta_index=2):
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
        return result + torch.exp(self.log_bg)

def training_loop(model, optimizer, n=1000):
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

def lbfgs_training(model,
                   loss_fn,
                   max_iter_adam=1000,
                   max_iter_lbfgs=1000,
                   conv_eps=1e-12,
                   learning_rate_adam=1e-3,
                   learning_rate_lbfgs=0.75):
    y = model.i_obs
    s = model.s_obs
    # first we do a few rounds of ADAM
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_adam)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    for ii in range(max_iter_adam):
        optimizer.zero_grad()
        preds = model()
        loss = loss_fn(preds / s, y / s)
        loss.backward()
        optimizer.step()
        scheduler.step()


    optimizer = torch.optim.LBFGS([model.weights],
                                  history_size=len(model.weights)*5+10,
                                  max_iter=max_iter_lbfgs,
                                  lr=learning_rate_lbfgs,
                                  line_search_fn='strong_wolfe')
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        output = model()
        loss = loss_fn(output/s, y/s)
        if loss.requires_grad:
            loss.backward()
        return loss

    history = []
    for ii in range(max_iter_lbfgs):
        before = model.weights.clone()
        history.append(before)
        optimizer.step(closure)
        after = model.weights
        sel = torch.isnan(after)
        if torch.sum(sel).item() > 0:
            jitter = (2.0*torch.rand(before.shape)-1.0)*0.05 + 1.0
            model.weights = nn.Parameter( before*jitter )
        else:
            delta = torch.mean(torch.abs(before-after)) / torch.mean( torch.abs(before))
            if delta < conv_eps:
                break

    with torch.no_grad():
        output = model()
        loss = loss_fn(output / s, y / s)
    return loss




class Kramer_interpolator(nn.Module):
    def __init__(self, radius, q_max=None, padding=None, Iqks=None):
        super().__init__()
        self.radius = radius
        self.radius = nn.Parameter(torch.Tensor([self.radius]))
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



def training(model,
                   loss_fn,
                   q_in,
                   i_obs,
                   s_obs,
                   max_iter_adam=100,
                   learning_rate=1e-3):
    x = q_in
    y = i_obs
    s = s_obs
    # first we do a few rounds of ADAM

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for ii in range(max_iter_adam):
        preds = model(x)
        loss = loss_fn(preds / s, y / s)
        print("Loss", loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        output = model(x)
        loss = F.mse_loss(output / s, y / s)
        if str(loss.item()) == "nan":
            print(model.weights)
    return loss






if __name__ == "__main__":

    #Define data
    q_obs_max = 0.55
    N_obs = 55
    q_obs = torch.Tensor(np.linspace(0.01, q_obs_max, N_obs))
    Iobs = 0
    radius = 30
    ws = 0.0
    sr = 0.01
    Iobs = gaussian_mixture_of_spheres.i_of_q(q_obs,radius,sr)
    Iobs1 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 1.5))
    Iobs2 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 1.5))
    Iobs3 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 1.5))
    Iobs4 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 1.5))
    Iobs5 = torch.abs(torch.normal(Iobs, ((q_obs ) ** 0.50) * Iobs / 1.5))
    Iobs = einops.rearrange([Iobs1, Iobs2, Iobs3, Iobs4, Iobs5], "N X -> N X")

    m = 1.01*torch.mean(Iobs, dim=0)+1e-5
    s = torch.std(Iobs, dim=0) / 2
    plt.plot(q_obs.numpy(), m, '.')
    plt.xlim(0, q_obs_max * 1.05)
    plt.yscale('log')
    plt.show()

    radius_range = range(15,75,1)

    scores = []
    k = []
    bgs = []
    for this_rad in radius_range:
        obj1 = Kramer_fit(q_obs,m,s, this_rad, padding=1)
        loss1 = lbfgs_training(obj1, F.l1_loss)
        bgs.append(obj1.log_bg.detach().numpy())
        print("AFTER", obj1.weights,  obj1.log_bg)
        k.append(len(obj1.weights))
        scores.append(loss1)
    plt.plot(radius_range, scores)
    #plt.yscale('log')
    plt.show()
    plt.plot(scores, bgs, '.')
    plt.xscale('log')
    plt.show()

    k = np.array(k)
    print(k)
    scores = np.array(scores)
    aic = 2*k + 350.0*np.log(scores) #- 350.0*np.log(350)
    bic = np.log(350)*k + 350.0*np.log(scores) #- 350.0*np.log(350)

    minaic = np.min(bic)

    plt.plot(radius_range, np.exp(-(bic-minaic)))
    plt.show()

    """obj1 = Kramer_fit(q_obs, m, s, radius, padding=5)
    Iqks = torch.exp(obj1.weights)

    obj2 = Kramer_interpolator(radius=radius, Iqks=Iqks)
    q = torch.Tensor(np.linspace(0,0.35,1000))
    tmp = obj2(q).detach()

    plt.plot(q_obs, m, '.')
    yyy1 = obj1().detach().numpy().flatten()
    plt.plot(q_obs, yyy1, '--', c='r')
    plt.plot(q, tmp, '-', c='g')

    plt.yscale('log')
    plt.show()



    training(obj2, F.l1_loss , q_obs, m, s, learning_rate=1e-2)
    tmp = obj2(q_obs)
    plt.plot(q_obs.numpy(), tmp.detach().flatten().numpy())
    plt.plot(q_obs.numpy(), m.numpy(), '.')
    plt.yscale('log')
    plt.show()
    """