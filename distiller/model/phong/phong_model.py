import torch
import torch.nn.functional as f

from distiller.model.phong.phong_base import PhongBase
from distiller.utils import const, config, funcs


class PhongModel(PhongBase):
    def __init__(self):
        super(PhongModel, self).__init__()

    def specular(self, light, normal, view):
        if config.USE_VEC:
            light2 = f.normalize(funcs.batch_scale(2 * funcs.batch_vec_dot(normal, light), normal) - light, p=2, dim=0)
            return torch.max(const.ZERO, funcs.batch_vec_dot(normal, light2))
        else:
            light2 = f.normalize(2 * normal.dot(light) * normal - light, p=2, dim=0)
            return torch.max(const.ZERO, view.dot(light2))
