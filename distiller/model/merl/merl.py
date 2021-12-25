import torch
import torch.nn.functional as f

from distiller.model.merl.merl_file_reader import Merl
from distiller.utils import config, funcs, const


class MerlModel:
    def __init__(self, name):
        self.__name = name
        self.__merl = Merl(f'{config.MERL_DIR}/{self.__name}.binary')

    # Rotate vector around arbitrary axis
    def __rotate_vector(self, vector, axis, alpha):
        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)
        return vector * cos_alpha + \
               axis * torch.dot(vector, axis) * (1 - cos_alpha) + \
               torch.cross(axis, vector) * sin_alpha

    def __coord(self, light, normal, view):
        # half = funcs.half(light, view)
        half = f.normalize(light + view, p=2, dim=0)
        theta_h = torch.acos(half[2])
        phi_h = torch.atan2(half[1], half[0])

        bi_normal = torch.tensor([0.0, 1.0, 0.0])
        tmp = self.__rotate_vector(view, normal, -phi_h)
        diff = self.__rotate_vector(tmp, bi_normal, -theta_h)

        theta_d = torch.acos(diff[2])
        phi_d = torch.atan2(diff[1], diff[0]) % const.PI

        return theta_h, theta_d, phi_d

    def __call__(self, *args, **kwargs):
        inputs = args[0]
        data_size = len(inputs)
        result = torch.empty(data_size, 3)
        for i in range(data_size):
            light = inputs[i][0]
            normal = torch.tensor([0.0, 0.0, 1.0])
            if config.USE_CUDA:
                normal = normal.cuda()
            view = inputs[i][1]
            theta_h, theta_d, phi_d = self.__coord(light, normal, view)
            result[i] = torch.tensor(self.__merl.eval_interp(theta_h, theta_d, phi_d))
        return result

        # light = inputs[:, 0:1, :].squeeze()
        # normal = torch.tensor([[0.0, 0.0, 1.0]])
        # if config.USE_CUDA:
        #     normal = normal.cuda()
        # view = inputs[:, 1:, :].squeeze()

    def __str__(self):
        return self.__name
