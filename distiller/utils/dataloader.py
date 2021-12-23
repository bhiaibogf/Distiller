"""
用于将数据封装成 DataLoader 的模块
"""
from torch.utils.data import DataLoader, TensorDataset

from distiller.utils import const


class Dataloader:
    def __init__(self, reader, batch_size):
        """
        :param reader: 用于获取数据的类
        :param batch_size: 一批数据的大小
        """
        self.__reader = reader
        self.__batch_size = batch_size

        x, y = reader.get_train_data()
        if const.USE_CUDA:
            x = x.cuda()
            y = y.cuda()
        self.__train_dataset = TensorDataset(x, y)

        x, y = reader.get_valid_data()
        if const.USE_CUDA:
            x = x.cuda()
            y = y.cuda()
        self.__valid_dataset = TensorDataset(x, y)

    def get_data(self):
        """
        获取未打包成批的数据

        :return: 未打包成批的数据
        """
        return self.__reader.get_valid_data()

    def get_train_dataloader(self):
        return DataLoader(self.__train_dataset, batch_size=self.__batch_size, shuffle=True)

    def get_valid_dataloader(self):
        return DataLoader(self.__valid_dataset, batch_size=self.__batch_size * 2)
