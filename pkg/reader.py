import torch
from torch.utils.data import DataLoader, TensorDataset


class Reader:
    def __init__(self, batch_size):
        self.__batch_size = batch_size
        self.get_data()

    def get_data(self):
        x = torch.linspace(-3, 3, 1024)
        y = torch.sin(x)
        self.__train_dataset = TensorDataset(x, y)
        x = torch.linspace(-3, 3, 1024)
        y = torch.sin(x)
        self.__valid_dataset = TensorDataset(x, y)
        return x, y

    def get_train_dataloader(self):
        return DataLoader(self.__train_dataset, batch_size=self.__batch_size, shuffle=True)

    def get_valid_dataloader(self):
        return DataLoader(self.__valid_dataset, batch_size=self.__batch_size * 2)
