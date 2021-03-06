import torch
import torch.nn as nn

from distiller.model.brdf_base import BrdfBase
from distiller.utils import const, funcs


class PhongBase(BrdfBase):
    def __init__(self):
        super(PhongBase, self).__init__()
        self._kd = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        self._ks = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        self._alpha = nn.Parameter(torch.tensor([10.0]))

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def specular(self, light, normal, view):
        pass

    def _shade(self, light, normal, view):
        intensity = 1 / const.PI / funcs.batch_vec_dot(normal, light)
        # diffuse
        l_d = funcs.batch_scale(intensity * torch.max(const.ZERO, funcs.batch_vec_dot(normal, light)),
                                self._kd)
        # specular
        l_s = funcs.batch_scale(intensity * torch.pow(self.specular(light, normal, view), self._alpha),
                                self._ks)
        return l_s + l_d

    def _eval(self, light, normal, view):
        intensity = 1 / const.PI / normal.dot(light)
        # diffuse
        l_d = self._kd * intensity * torch.max(const.ZERO, normal.dot(light))
        # specular
        l_s = self._ks * intensity * torch.pow(self.specular(light, normal, view), self._alpha)
        return l_s + l_d

    def clamp_(self):
        self._kd.data.clamp_(0, 1)
        self._ks.data.clamp_(0, 1)

    # def __str__(self):
    #     alpha = self.__alpha.data.item()
    #     kd = self.__kd.data.tolist()
    #     ks = self.__ks.data.tolist()
    #     return 'Ns {}\nkd {} {} {}\nks {} {} {}\n'.format(alpha, *kd, *ks)
