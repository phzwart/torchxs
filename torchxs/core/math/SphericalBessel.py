import torch

def j0(x, eps=1e-7):
    top = torch.sin(x)
    bottom = torch.clone(x)
    sel = torch.abs(x) < eps
    top[sel] = 1.0 - (x[sel]*x[sel])/6.0
    bottom[sel] = 1.0
    return top / bottom

def j1(x, eps=1e-7):

    top1 = torch.sin(x)
    bottom1 = torch.clone(x)**2.0

    top2 = torch.cos(x)
    bottom2 = torch.clone(x)

    sel = torch.abs(x) < eps

    top1[sel] = x[sel]/3.0 - ((x[sel])**3.0)/30.0 + ((x[sel])**5.0)/840.0
    bottom1[sel]=1.0
    top2[sel]=0.0
    bottom2[sel]=1.0

    result = top1/bottom1 - top2/bottom2
    return result



