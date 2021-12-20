"""
Equatiuons taken from

https://www.ncnr.nist.gov/staff/hammouda/distance_learning/chapter_27.pdf
"""

import torch
import einops
from torchxs.core.math import SphericalBessel
from torchxs.core.shapes import sphere
import numpy as np
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt
import plotly.express as px

def i_of_q(q_tensor, radius, sigma=1.0, N=10, eps=1e-1):
    x,w = hermgauss(N)
    x = torch.Tensor(x*sigma+radius)
    w = torch.Tensor(w)
    tmp = sphere.i_of_q(q_tensor,x)
    tmp = torch.mul(tmp,w)
    result = torch.sum(tmp, axis=1) / torch.sum(w)
    return result

def p_of_r(r_tensor, radius, eps=1e-7):
    return None

def gamma_of_r(r_tensor, radius, eps=1e-7):
    return None
