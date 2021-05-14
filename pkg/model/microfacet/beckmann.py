import math

import torch

from pkg.model.microfacet.base import MicrofacetBase, quick_pow


class BeckmannModel(MicrofacetBase):
    def __init__(self):
        super(BeckmannModel, self).__init__()

    def d(self, nh):
        alpha_2 = quick_pow(self._alpha, 2)
        nh_2 = quick_pow(nh, 2)
        return torch.exp((nh_2 - 1) / (alpha_2 * nh_2)) / math.pi / alpha_2 / quick_pow(nh, 4)

    def g1(self, nv):
        c = self._c(nv)
        c2 = quick_pow(c, 2)
        if c < 1.6:
            return (3.535 * c + 2.181 * c2) / (1.0 + 2.276 * c + 2.577 * c2)
        else:
            return 1

    def g(self, light, normal, view):
        return self.g1(normal.dot(view)) * self.g1(normal.dot(light))
