import torch
import torch.nn.functional as f

from pkg.model.phong.phong_base import PhongBase


class PhongModel(PhongBase):
    def __init__(self):
        super(PhongModel, self).__init__()

    def specular(self, light, normal, view):
        light2 = f.normalize(2 * (torch.dot(light, normal)) * normal - light, p=2, dim=0)
        return torch.max(torch.zeros(1), torch.dot(view, light2))
