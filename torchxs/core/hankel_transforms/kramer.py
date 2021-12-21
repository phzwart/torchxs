import torch
import numpy as np
from torchxs.core.math import SphericalBessel
from torchxs.core.shapes import gaussian_mixture_of_spheres

import matplotlib.pyplot as plt


class Kramer(object):
    def __init__(self, q_tensor, i_obs, s_obs, radius, padding=0):
        self.q_tensor = q_tensor

        self.radius = radius
        self.padding = padding

        self.i_obs = i_obs
        self.s_obs = s_obs

        self.max_k = int(torch.max(self.q_tensor * self.radius * 2.0 / np.pi) + 1) + self.padding
        self.alphas = torch.Tensor(np.arange(1,self.max_k+1)*np.pi)
        self.qks = self.alphas / ( 2.0 * self.radius)

        self.Skq = self.intensity_basis_functions()
        self.Iqks = self.initial_guess()

    def intensity_basis_functions(self):
        t = self.q_tensor * self.radius * 2.0
        tmp_top = torch.outer(self.alphas/SphericalBessel.j1(self.alphas), SphericalBessel.j0(t) )
        tmp_bottom = torch.outer( self.alphas**2.0, torch.ones(t.shape) )
        tmp_bottom = tmp_bottom - t*t
        result = 2.0*tmp_top/tmp_bottom
        return result.T

    def initial_guess(self):
        Iqks = []
        for qk in self.qks:
            if qk < torch.max(self.q_tensor):
                this_indx = torch.argmin(torch.abs(self.q_tensor - qk))
                Iqks.append(self.i_obs[this_indx])
            else:
                Iqks.append(self.i_obs[-1])
        Iqks = torch.Tensor(Iqks)
        return Iqks

    def series(self, x):
        result = torch.matmul( self.Skq,x.unsqueeze(dim=1))
        return result









if __name__ == "__main__":
    radius = 40.0
    q = torch.Tensor(np.linspace(0,0.45,150))
    m = gaussian_mixture_of_spheres.i_of_q(q, radius, 1)
    s = m/1000.0

    obj = Kramer(q,m,s, radius+10)
    xx = obj.Iqks
    print(xx)
    yy = obj.series(xx)
    print(yy.shape)

    import plotly.express as px
    px.imshow(obj.Skq.T).show()
    plt.plot(q,m, '.')
    plt.plot(q, yy.numpy().flatten(), '--')
    plt.yscale('log')
    plt.show()













