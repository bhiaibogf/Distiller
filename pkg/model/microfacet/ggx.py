import math

import torch

from pkg.model.microfacet.base import MicrofacetBase, quick_pow


class GgxModel(MicrofacetBase):
    def __init__(self):
        super(GgxModel, self).__init__()

    def d(self, nh):
        alpha_2 = quick_pow(self._alpha, 2)
        return alpha_2 / math.pi / quick_pow(torch.lerp(torch.tensor([1.0]), alpha_2, quick_pow(nh, 2)), 1)

    def g1(self, nv):
        return 2 * nv / (nv + torch.sqrt(torch.lerp(quick_pow(self._alpha, 2), torch.tensor([1.0]), quick_pow(nv, 2))))

    def g(self, light, normal, view):
        return self.g1(normal.dot(view)) * self.g1(normal.dot(light))
