import math

import torch
import torch.nn as nn


class PhongBase(nn.Module):
    def __init__(self):
        super(PhongBase, self).__init__()
        self.__kd = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.__ks = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.__alpha = nn.Parameter(torch.tensor(64.0))

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)

    def specular(self, light, normal, view):
        return 1

    def forward(self, inputs):
        data_size = len(inputs)
        ls = torch.empty(data_size, 3)
        for i in range(data_size):
            intensity = 1 / math.pi
            light = inputs[i][0]
            normal = inputs[i][1]
            view = inputs[i][2]
            # diffuse
            l_d = self.__kd * intensity * torch.max(torch.zeros(1), torch.dot(light, normal))
            # specular
            l_s = self.__ks * intensity * torch.pow(self.specular(light, normal, view), self.__alpha)
            ls[i] = l_s + l_d
        return ls

    def clamp_(self):
        self.__kd.data.clamp_(0, 1)
        self.__ks.data.clamp_(0, 1)
        self.__alpha.data.clamp_(1, 1024)

    def __str__(self):
        return 'Ns {}\nkd {}\nks {}'.format(self.__alpha.data.item(), self.__kd.data.tolist(), self.__ks.data.tolist())
