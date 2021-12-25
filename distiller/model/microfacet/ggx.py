import torch

from distiller.model.microfacet.microfacet_base import MicrofacetBase
from distiller.utils import const, config, funcs


class GgxModel(MicrofacetBase):
    def __init__(self):
        super(GgxModel, self).__init__()

    def d(self, cos_nm):
        alpha_2 = funcs.sqr(self._alpha)
        return alpha_2 / const.PI / funcs.sqr(torch.lerp(const.ONE, alpha_2, funcs.sqr(cos_nm)))

    def g1(self, cos_ns):
        return 2 * cos_ns / (cos_ns + torch.sqrt(
            torch.lerp(funcs.sqr(self._alpha), const.ONE, funcs.sqr(cos_ns))))

    def g(self, light, normal, view):
        if config.USE_VEC:
            return self.g1(funcs.batch_vec_dot(normal, view)) * self.g1(funcs.batch_vec_dot(normal, light))
        else:
            return self.g1(normal.dot(view)) * self.g1(normal.dot(light))
