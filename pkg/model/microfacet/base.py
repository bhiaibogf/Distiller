import torch
import torch.nn as nn
import torch.nn.functional as f

from pkg.model.utils import quick_pow, mon2lin


class MicrofacetBase(nn.Module):
    def __init__(self):
        super(MicrofacetBase, self).__init__()
        self._base_color = nn.Parameter(torch.tensor([0.8, 0.8, 0.8]))
        self._alpha = nn.Parameter(torch.tensor([0.25]))
        self._eta = nn.Parameter(torch.tensor([1.45]))

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def d(self, cos_nh):
        return 0

    def _schlick(self, cos_hv):
        f_0 = quick_pow((self._eta - 1) / (self._eta + 1), 2)
        return torch.lerp(f_0, torch.tensor([1.0]), quick_pow(1 - cos_hv, 5))

    def _cook_torrance(self, cos_hv):
        c = cos_hv
        g = quick_pow(self._eta, 2) + quick_pow(c, 2) - 1
        if g > 0:
            g = torch.sqrt(g)
            a = (g - c) / (g + c)
            b = (c * (g + c) - 1) / (c * (g - c) + 1)
            return 0.5 * quick_pow(a, 2) * (1 + quick_pow(b, 2))
        return 1

    def f(self, cos_hv):
        return self._cook_torrance(cos_hv)

    def _c(self, cos_nv):
        return cos_nv / self._alpha / torch.sqrt(1 - quick_pow(cos_nv, 2))

    def g(self, light, normal, view):
        return normal.dot(light) * normal.dot(view)

    def forward(self, inputs):
        data_size = len(inputs)
        ls = torch.empty(data_size, 3)
        for i in range(data_size):
            light = inputs[i][0]
            normal = torch.tensor([0.0, 0.0, 1.0])
            view = inputs[i][1]
            half = f.normalize(light + view, p=2, dim=0)
            ls[i] = mon2lin(self._base_color)
            ls[i] *= self.d(normal.dot(half)) * self.g(light, normal, view) * self.f(half.dot(view))
            ls[i] /= 4 * normal.dot(light) * normal.dot(view)
        return ls

    def clamp_(self):
        self._alpha.data.clamp_(0, 1)
        self._eta.data.clamp_(1, 5)

    def __str__(self):
        return 'basecolor = ({}, {}, {})\nalpha = {}, eta = {}'.format(
            *self._base_color.tolist(), self._alpha.data.item(), self._eta.data.item())
