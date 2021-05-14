import torch
import torch.nn.functional as f

from pkg.model.phong.phong_base import PhongBase


class BlinnPhongModel(PhongBase):
    def __init__(self):
        super(BlinnPhongModel, self).__init__()

    def specular(self, light, normal, view):
        half = f.normalize(light + view, p=2, dim=0)
        return torch.max(torch.zeros(1), torch.dot(normal, half))
