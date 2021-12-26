from abc import ABCMeta, abstractmethod

import pandas as pd
import torch

from distiller.utils import config
from distiller.utils.sampler import Sampler


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


class ModelReader(Reader):
    def __init__(self, train_data_size, valid_data_size, model):
        super(ModelReader, self).__init__(train_data_size, valid_data_size)
        self.__model = model

    def _get_data(self, inputs):
        return self.__model(inputs).detach()

    def get_train_data(self):
        light = Sampler.cos_hemisphere(self._train_data_size)
        view = Sampler.hemisphere(self._train_data_size)
        inputs = torch.stack((light, view), 1)
        assert inputs.shape == (self._train_data_size, 2, 3)
        if config.USE_CUDA:
            inputs = inputs.cuda()

        return inputs, self._get_data(inputs)

    def get_valid_data(self):
        light = Sampler.cos_hemisphere(self._valid_data_size)
        view = Sampler.hemisphere(self._valid_data_size)
        inputs = torch.stack((light, view), 1)
        if config.USE_CUDA:
            inputs = inputs.cuda()

        return inputs, self._get_data(inputs)


class BsdfReader(Reader):
    def __init__(self, filename, train_data_size, valid_data_size):
        super(BsdfReader, self).__init__(train_data_size, valid_data_size)
        with open(filename) as file:
            self.__lines = file.readlines()

    def __read_file(self, data_size):
        x = torch.empty(data_size, 2, 3)
        y = torch.empty(data_size, 3)
        for i in range(data_size):
            line = self.__lines[i].replace('(', '').replace(')', '').split('|')
            x[i][0] = torch.tensor([float(x) for x in line[0].split(' ')])
            x[i][1] = torch.tensor([float(x) for x in line[1].split(' ')])
            y[i] = torch.tensor([float(x) for x in line[2].split(' ')])
        return x, y

    def get_train_data(self):
        return self.__read_file(self._train_data_size)

    def get_valid_data(self):
        return self.__read_file(self._valid_data_size)


class BsdfReader2(Reader):
    def __init__(self, filename, train_data_size, valid_data_size):
        super(BsdfReader2, self).__init__(train_data_size, valid_data_size)
        self.__df = pd.read_csv(
            filename, header=None,
            sep=r'[\(\)|\ ]', engine='python',
            usecols=[1, 2, 3, 6, 7, 8, 11, 12, 13]
        )
        # print(self.__df)

    def __get_data(self, data_size):
        df = self.__df.sample(data_size)
        x = df.iloc[:, [0, 1, 2, 3, 4, 5]].to_numpy()
        y = df.iloc[:, [6, 7, 8]].to_numpy()
        return torch.tensor(x, dtype=torch.float).view(-1, 2, 3), torch.tensor(y, dtype=torch.float)

    def get_train_data(self):
        return self.__get_data(self._train_data_size)

    def get_valid_data(self):
        return self.__get_data(self._valid_data_size)


if __name__ == '__main__':
    reader = BsdfReader2('../../BSDF/alum-bronze.txt', 5, 3)
    print(reader.get_train_data())
