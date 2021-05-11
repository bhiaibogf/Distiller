import math

import torch

from pkg.model.microfacet.base import MicrofacetBase


class GgxModel(MicrofacetBase):
    def __init__(self):
        super(GgxModel, self).__init__()

    def d(self, nh):
        alpha_2 = self._alpha ** 2
        return alpha_2 / math.pi / (nh ** 2 * (alpha_2 - 1) + 1) ** 2

    def g1(self, nv):
        c2 = self._c(nv) ** 2
        return 2 / (1 + torch.sqrt(1 / c2 + 1))

    def g(self, light, normal, view):
        return self.g1(normal.dot(view)) * self.g1(normal.dot(light))
