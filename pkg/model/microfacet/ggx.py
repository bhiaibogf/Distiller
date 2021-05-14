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
        return 2 * nv / (nv + torch.sqrt(torch.lerp(self._alpha ** 2, torch.tensor([1.0]), nv ** 2)))

    def g(self, light, normal, view):
        return self.g1(normal.dot(view)) * self.g1(normal.dot(light))
