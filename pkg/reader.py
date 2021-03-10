import torch


class Reader:
    def __init__(self):
        pass

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
