import torch

from distiller.model.microfacet.microfacet_base import MicrofacetBase
from distiller.utils import const, funcs


class BeckmannModel(MicrofacetBase):
    def __init__(self):
        super(BeckmannModel, self).__init__()

    def d(self, cos_nm):
        alpha_2 = funcs.sqr(self._alpha)
        nh_2 = funcs.sqr(cos_nm)
        numer = torch.exp((nh_2 - 1) / torch.max(const.POINT_OO_ONE, alpha_2 * nh_2))
        denom = const.PI * alpha_2 * funcs.quick_pow(cos_nm, 4)
        return numer / torch.max(const.POINT_OO_ONE, denom)

    def g1(self, cos_ns):
        c = self._c(cos_ns).clamp(0, 1.6)
        c2 = funcs.sqr(c)
        return (3.535 * c + 2.181 * c2) / (1.0 + 2.276 * c + 2.577 * c2)

    def g(self, light, normal, view):
        if const.USE_VEC:
            return self.g1(funcs.batch_vec_dot(normal, view)) * self.g1(funcs.batch_vec_dot(normal, light))
        else:
            return self.g1(normal.dot(view)) * self.g1(normal.dot(light))
