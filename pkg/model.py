import torch
import torch.nn as nn
import torch.nn.functional as F


class PolynomialModel(nn.Module):
    def __init__(self, dim):
        super(PolynomialModel, self).__init__()
        self.__dim = dim
        self.__w = nn.Parameter(torch.rand(dim))
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        inputs = torch.ones(self.__dim, x.shape[0])
        for i in range(1, self.__dim):
            inputs[i] = inputs[i - 1] * x
        return torch.matmul(self.__w, inputs)


class BlinnPhongModel(nn.Module):
    def __init__(self):
        super(BlinnPhongModel, self).__init__()
        # self.__ka = nn.Parameter(torch.rand(3))
        self.__kd = nn.Parameter(torch.rand(3))
        self.__ks = nn.Parameter(torch.rand(3))
        self.__p = nn.Parameter(torch.rand(3))

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs):
        bat_size = inputs.shape[0]
        ls = torch.ones(bat_size, 3)
        for i in range(bat_size):
            intensity = 1
            light = inputs[i][0]
            normal = inputs[i][1]
            view = inputs[i][2]
            # diffuse
            l_d = self.__kd * intensity * torch.max(torch.zeros(1), torch.dot(light, normal))
            # specular
            half = F.normalize(light + view, p=2, dim=0)
            l_s = self.__ks * intensity * torch.pow(torch.max(torch.zeros(1), torch.dot(normal, half)), self.__p[0])
            ls[i] = l_s + l_d
        return ls

    def __str__(self):
        return 'kd={}\nks={}\np={}'.format(self.__kd.data, self.__ks.data, self.__p.data)
