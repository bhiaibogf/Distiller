import torch

from distiller.model.microfacet.microfacet_base import MicrofacetBase
from distiller.utils import const, funcs


class BeckmannModel(MicrofacetBase):
    def __init__(self):
        super(BeckmannModel, self).__init__()

    def d(self, cos_nm):
        if cos_nm < 0:
            return const.ZEROS
        alpha_2 = funcs.sqr(self._alpha)
        nh_2 = funcs.sqr(cos_nm)
        return torch.exp((nh_2 - 1) / (alpha_2 * nh_2)) / const.PI / alpha_2 / funcs.quick_pow(cos_nm, 4)

    def g1(self, cos_ns):
        c = self._c(cos_ns)
        c2 = funcs.sqr(c)
        if c < 1.6:
            return (3.535 * c + 2.181 * c2) / (1.0 + 2.276 * c + 2.577 * c2)
        else:
            return 1

    def g(self, light, normal, view):
        return self.g1(normal.dot(view)) * self.g1(normal.dot(light))
