from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F


class Reader(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_train_data():
        pass

    @staticmethod
    @abstractmethod
    def get_valid_data():
        pass


class PolynomialReader(Reader):
    @staticmethod
    def get_train_data():
        x = torch.linspace(-3, 3, 1024)
        y = torch.sin(x)
        return x, y

    @staticmethod
    def get_valid_data():
        x = torch.linspace(-3, 3, 1024)
        y = torch.sin(x)
        return x, y


class BlinnPhongReader(Reader):
    def __init__(self):
        self.__kd = torch.tensor([0.5, 0.7, 0.2])
        self.__ks = torch.tensor([0.3, 0.2, 0.1])
        self.__p = torch.tensor([12, 0, 0])

    def get_train_data(self):
        data_size = 1024
        inputs = F.normalize(torch.rand(data_size, 3, 3), p=2, dim=2)
        ls = torch.ones(data_size, 3)
        for i in range(data_size):
            intensity = 1
            light = inputs[i][0]
            normal = inputs[i][1]
            view = inputs[i][2]
            # diffuse
            l_d = self.__kd * intensity * torch.max(torch.zeros(1), torch.dot(light, normal))
            # specular
            half = F.normalize(light + view, p=2,dim=0)
            l_s = self.__ks * intensity * torch.pow(torch.max(torch.zeros(1), torch.dot(normal, half)), self.__p)
            ls[i] = l_s + l_d
        return inputs, ls

    def get_valid_data(self):
        return self.get_train_data()
