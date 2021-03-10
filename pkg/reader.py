from abc import ABCMeta, abstractmethod

import torch


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
