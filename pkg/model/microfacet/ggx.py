import torch

from pkg.model.microfacet.base import MicrofacetBase, sqr, PI


class GgxModel(MicrofacetBase):
    def __init__(self):
        super(GgxModel, self).__init__()

    def d(self, cos_nm):
        alpha_2 = sqr(self._alpha)
        return alpha_2 / PI / sqr(torch.lerp(torch.tensor([1.0]), alpha_2, sqr(cos_nm)))

    def g1(self, cos_ns):
        return 2 * cos_ns / (cos_ns + torch.sqrt(
            torch.lerp(sqr(self._alpha), torch.tensor([1.0]), sqr(cos_ns))))

    def g(self, light, normal, view):
        return self.g1(normal.dot(view)) * self.g1(normal.dot(light))
