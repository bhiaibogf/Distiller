"""
用于训练模型的模块
"""
import math

import matplotlib.pyplot as plt
import torch


class Trainer:
    """
    训练模型用的类
    """

    def __init__(self, model, train_dataloader, valid_dataloader):
        """
        :param model: 需要训练的模型
        :param train_dataloader: 训练集
        :param valid_dataloader: 测试集
        """
        self.__model = model
        self.__train_dataloader = train_dataloader
        self.__valid_dataloader = valid_dataloader

        self.__losses = []
        self.__accuracies = []

    def __plot(self):
        fig, axes_loss = plt.subplots()
        fig.suptitle('training steps')
        fig.subplots_adjust(right=0.85)

        axes_loss.plot(
            range(len(self.__losses)), self.__losses,
            label='loss',
            color='green'
        )
        axes_loss.set_xlabel('epochs')
        axes_loss.set_ylabel('loss', color='green')

        if self.__accuracies:
            axes_accuracy = axes_loss.twinx()
            axes_accuracy.plot(
                range(len(self.__accuracies)), self.__accuracies,
                label='accuracy',
                color='red'
            )
            axes_accuracy.set_ylabel('accuracy', color='red')

            fig.legend(loc='center right',
                       bbox_to_anchor=(1, 0.5), bbox_transform=axes_loss.transAxes)

        plt.savefig('img/' + self.__model.__class__.__name__ + '.png')
        plt.draw()
        plt.show()

    @staticmethod
    def __cos(x, y):
        light_z = x[:, 0:1, -1:]
        cos = light_z.squeeze()
        return torch.einsum('i,ij->ij', cos, y)

    def __loss(self, x, y, need_backward=False):
        y_pre = self.__model(x)
        loss = self.__model.loss_function(y_pre, y)
        # loss = self.__model.loss_function(self.__cos(x, y_pre), self.__cos(x, y))
        if need_backward:
            loss.backward()
            # 检查是否发生梯度爆炸
            # for p in self.__model.parameters():
            #     if torch.isnan(p.grad[0]):
            #         print(self.__model)
            #         raise Exception

            self.__model.optimizer.step()
            self.__model.clamp_()
            self.__model.optimizer.zero_grad()
        return loss.data.item()

    def train(self, epochs):
        """
        使用训练集与测试机训练模型

        :param epochs: 训练遍数
        :return: None
        """
        for epoch in range(epochs):
            # 训练
            for x, y in self.__train_dataloader:
                self.__loss(x, y, True)

            # 测试
            losses, cnt = 0, 0
            max_brdf = 0
            with torch.no_grad():
                for x, y in self.__valid_dataloader:
                    losses += self.__loss(x, y) * len(y)
                    cnt += len(y)
                    max_brdf = max(max_brdf, torch.max(y))
                    # max_brdf = max(max_brdf, torch.max(self.__cos(x, y)))
            loss = losses / cnt
            psnr = 10 * math.log10(max_brdf * max_brdf / loss)
            print('epoch {} :\nloss : {}\npsnr : {}({})'.format(epoch, loss, psnr, max_brdf))
            self.__losses.append(loss)
            # self.__accuracies.append(1 - loss)
            print(self.__model)
            if self.__model.lr:
                self.__model.lr.step()
        self.__plot()

    def pre(self, x):
        """
        使用模型预测结果

        :param x: 模型输入
        :return: 预测结果
        """
        return self.__model(x).detach()
