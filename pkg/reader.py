from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as f


class Reader(metaclass=ABCMeta):
    def __init__(self, train_data_size, valid_data_size):
        self._train_data_size = train_data_size
        self._valid_data_size = valid_data_size

    @abstractmethod
    def get_train_data(self):
        pass

    @abstractmethod
    def get_valid_data(self):
        pass


class PolynomialReader(Reader):
    def get_train_data(self):
        x = torch.linspace(-3, 3, self._train_data_size)
        y = torch.sin(x)
        return x, y

    def get_valid_data(self):
        x = torch.linspace(-3, 3, self._valid_data_size)
        y = torch.sin(x)
        return x, y


class BlinnPhongReader(Reader):
    def __init__(self, train_data_size, valid_data_size):
        super(BlinnPhongReader, self).__init__(train_data_size, valid_data_size)
        self.__kd = torch.tensor([0.3, 0.2, 0.1])
        self.__ks = torch.tensor([0.5, 0.9, 0.1])
        self.__p = torch.tensor(8.0)

    def __get_data(self, inputs):
        data_size = len(inputs)
        ls = torch.empty(data_size, 3)
        for i in range(data_size):
            intensity = 1
            light = inputs[i][0]
            normal = inputs[i][1]
            view = inputs[i][2]
            # diffuse
            l_d = self.__kd * intensity * torch.max(torch.zeros(1), torch.dot(light, normal))
            # specular
            half = f.normalize(light + view, p=2, dim=0)
            l_s = self.__ks * intensity * torch.pow(torch.max(torch.zeros(1), torch.dot(normal, half)), self.__p)
            ls[i] = l_s + l_d
        return ls

    def get_train_data(self):
        inputs = f.normalize(torch.randn(self._train_data_size, 3, 3), p=2, dim=2)
        return inputs, self.__get_data(inputs)

    def get_valid_data(self):
        inputs = f.normalize(torch.randn(self._valid_data_size, 3, 3), p=2, dim=2)
        return inputs, self.__get_data(inputs)


class BsdfReader(Reader):
    def __init__(self, filename, train_data_size, valid_data_size):
        super(BsdfReader, self).__init__(train_data_size, valid_data_size)
        with open(filename) as file:
            self.lines = file.readlines()

    def __read_file(self, data_size):
        x = torch.empty(data_size, 3, 3)
        y = torch.empty(data_size, 3)
        for i in range(data_size):
            line = self.lines[i].replace('(', '').replace(')', '').split('|')
            x[i][0] = torch.tensor([float(x) for x in line[0].split(' ')])
            x[i][1] = torch.tensor([0, 0, 1])
            x[i][2] = torch.tensor([float(x) for x in line[1].split(' ')])
            y[i] = torch.tensor([float(x) for x in line[2].split(' ')])
        return x, y

    def get_train_data(self):
        return self.__read_file(self._train_data_size)

    def get_valid_data(self):
        return self.__read_file(self._valid_data_size)


if __name__ == '__main__':
    reader = BsdfReader(2, 1)
    x, y = reader.get_train_data()
