import numpy as np
import torch

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
        if config.USE_VEC:
            a = funcs.batch_scale_batch(cos_alpha, vector)
            b = funcs.batch_scale_batch(1 - cos_alpha, funcs.batch_scale(funcs.batch_vec_dot(vector, axis), axis))
            if config.USE_CUDA:
                c = funcs.batch_scale_batch(sin_alpha,
                                            torch.tensor(np.cross(axis.to('cpu').numpy(), vector.to('cpu').numpy())
                                                         , device='cuda'))
            else:
                c = funcs.batch_scale_batch(sin_alpha, torch.tensor(np.cross(axis.numpy(), vector.numpy())))
        else:
            a = vector * cos_alpha
            b = axis * torch.dot(vector, axis) * (1 - cos_alpha)
            c = torch.cross(axis, vector) * sin_alpha
        return a + b + c

    def __coord(self, light, normal, view):
        half = funcs.half(light, view)
        if config.USE_VEC:
            theta_h = torch.acos(half[:, 2])
            phi_h = torch.atan2(half[:, 1], half[:, 0])
        else:
            theta_h = torch.acos(half[2])
            phi_h = torch.atan2(half[1], half[0])

        bi_normal = torch.tensor([0.0, 1.0, 0.0])
        if config.USE_VEC:
            bi_normal = bi_normal.unsqueeze(0)
        if config.USE_CUDA:
            bi_normal = bi_normal.cuda()

        tmp = self.__rotate_vector(view, normal, -phi_h)
        diff = self.__rotate_vector(tmp, bi_normal, -theta_h)

        if config.USE_VEC:
            theta_d = torch.acos(diff[:, 2])
            phi_d = torch.atan2(diff[:, 1], diff[:, 0]) % const.PI
        else:
            theta_d = torch.acos(diff[2])
            phi_d = torch.atan2(diff[1], diff[0]) % const.PI

        return theta_h, theta_d, phi_d

    def __handel(self, inputs):
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

    def __handel_vec(self, inputs):
        light = inputs[:, 0, :].squeeze()
        normal = torch.tensor([[0.0, 0.0, 1.0]])
        if config.USE_CUDA:
            normal = normal.cuda()
        view = inputs[:, 1, :].squeeze()
        theta_h, theta_d, phi_d = self.__coord(light, normal, view)

        data_size = len(inputs)
        result = torch.empty(data_size, 3)
        for i in range(data_size):
            result[i] = torch.tensor(self.__merl.eval_interp(theta_h[i], theta_d[i], phi_d[i]))
        return result

    def __call__(self, *args, **kwargs):
        inputs = args[0]
        if config.USE_VEC:
            return self.__handel_vec(inputs)
        else:
            return self.__handel(inputs)

    def __str__(self):
        return self.__name
