import math

import torch

from pkg.model.microfacet.base import MicrofacetBase, sqr


class GgxModel(MicrofacetBase):
    def __init__(self):
        super(GgxModel, self).__init__()

    def d(self, cos_nh):
        alpha_2 = sqr(self._alpha)
        return alpha_2 / math.pi / sqr(torch.lerp(torch.tensor([1.0]), alpha_2, sqr(cos_nh)))

    def g1(self, cos_nv):
        return 2 * cos_nv / (cos_nv + torch.sqrt(
            torch.lerp(sqr(self._alpha), torch.tensor([1.0]), sqr(cos_nv))))

    def g(self, light, normal, view):
        return self.g1(normal.dot(view)) * self.g1(normal.dot(light))
