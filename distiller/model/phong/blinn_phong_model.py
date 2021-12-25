import torch

from distiller.model.phong.phong_base import PhongBase
from distiller.utils import const, config, funcs


class BlinnPhongModel(PhongBase):
    def __init__(self):
        super(BlinnPhongModel, self).__init__()

    def specular(self, light, normal, view):
        half = funcs.half(light, view)
        if config.USE_VEC:
            return torch.max(const.ZERO, funcs.batch_vec_dot(normal, half))
        else:
            return torch.max(const.ZERO, normal.dot(half))
