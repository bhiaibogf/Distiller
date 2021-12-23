import torch
import torch.nn as nn
import torch.nn.functional as f

from distiller.model.brdf_base import BrdfBase
from distiller.utils import const, funcs


class MicrofacetBase(BrdfBase):
    def __init__(self):
        super(MicrofacetBase, self).__init__()

        self._diffuse_color = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))

        self._base_color = nn.Parameter(torch.tensor([0.8, 0.8, 0.8]))
        self._alpha = nn.Parameter(torch.tensor([0.5]))
        self._eta = nn.Parameter(torch.tensor([1.45]))

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

    def d(self, cos_nh):
        return 0

    def _schlick(self, cos_hv):
        f_0 = funcs.sqr((self._eta - 1) / (self._eta + 1))
        return torch.lerp(f_0, const.ONES, funcs.quick_pow(1 - cos_hv, 5))

    def _cook_torrance(self, cos_hv):
        c = cos_hv
        g = funcs.sqr(self._eta) + funcs.sqr(c) - 1
        if g > 0:
            g = torch.sqrt(g)
            a = (g - c) / (g + c)
            b = (c * (g + c) - 1) / (c * (g - c) + 1)
            return 0.5 * funcs.sqr(a) * (1 + funcs.sqr(b))
        return 1

    def f(self, cos_hv):
        return self._schlick(cos_hv)

    def _c(self, cos_nv):
        return cos_nv / self._alpha / torch.sqrt(1 - funcs.sqr(cos_nv))

    def g(self, light, normal, view):
        return normal.dot(light) * normal.dot(view)

    def _eval(self, light, normal, view):
        half = f.normalize(light + view, p=2, dim=0)
        ls = funcs.mon2lin(self._base_color)
        ls *= self.d(normal.dot(half)) * self.g(light, normal, view) * self.f(half.dot(view))
        ls /= 4 * normal.dot(light) * normal.dot(view)
        ls += funcs.mon2lin(self._diffuse_color) / const.PI
        return ls

    def clamp_(self):
        self._diffuse_color.data.clamp_(0, 1)

        self._alpha.data.clamp_(0, 1)
        self._eta.data.clamp_(1, 10)
        self._base_color.data.clamp_(0, 1)
