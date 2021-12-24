import torch
import torch.nn.functional as f

from distiller.model.phong.phong_base import PhongBase
from distiller.utils import const, funcs


class BlinnPhongModel(PhongBase):
    def __init__(self):
        super(BlinnPhongModel, self).__init__()

    def specular(self, light, normal, view):
        if const.USE_VEC:
            half = f.normalize(light + view, p=2, dim=1)
            return torch.max(const.ZERO, funcs.batch_vec_dot(normal, half))
        else:
            half = f.normalize(light + view, p=2, dim=0)
            return torch.max(const.ZERO, normal.dot(half))
