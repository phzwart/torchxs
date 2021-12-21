"""
Equatiuons taken from

https://www.ncnr.nist.gov/staff/hammouda/distance_learning/chapter_27.pdf
"""

import torch
import einops
from torchxs.core.math import SphericalBessel

def i_of_q(q_tensor, radius, eps=1e-1):
    qr_bottom = torch.outer(q_tensor, radius)
    #print(qr_bottom)
    qr_top = 3.0*SphericalBessel.j1(qr_bottom, eps)
    sel = torch.abs(qr_bottom) < eps
    qr_top = qr_top**2.0
    qr_bottom = qr_bottom**2.0
    qr_top[sel] = 1.0 - qr_bottom[sel]/5.0 +3.0*(qr_bottom[sel]**2.0)/175 - 4.0*(qr_bottom[sel]**3.0)/4725
    qr_bottom[sel]=1.0
    return qr_top/qr_bottom

def p_of_r(r_tensor, radius, eps=1e-7):
    rR = r_tensor/radius
    result = 3.0 * rR * rR * (1 - 0.75 * rR + (rR ** 3) / 16.0 )
    return result

def gamma_of_r(r_tensor, radius, eps=1e-7):
    rR = r_tensor / radius
    result = 3.0 * (1 - 0.75 * rR + (rR ** 3) / 16.0)
    return result