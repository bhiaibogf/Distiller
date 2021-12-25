"""
用于训练模型的模块
"""
import math
import os

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
        :param valid_dataloader: 验证集
        """
        self.__model = model
        self.__train_dataloader = train_dataloader
        self.__valid_dataloader = valid_dataloader

        self.__losses = []
        self.__accuracies = []
        self.__psnrs = []
        self.__brdf_max = 0

    def plot(self, filename):
        """
        画训练时的 loss/psnr 变化
        """
        fig, axes_loss = plt.subplots()
        fig.suptitle('training steps')
        fig.subplots_adjust(right=0.85)

        # loss
        axes_loss.plot(
            range(len(self.__losses)), self.__losses,
            label='loss',
            color='green'
        )
        axes_loss.set_xlabel('epochs')
        axes_loss.set_ylabel('loss', color='green')

        # acc
        if self.__accuracies:
            axes_accuracy = axes_loss.twinx()
            axes_accuracy.plot(
                range(len(self.__accuracies)), self.__accuracies,
                label='accuracy',
                color='red'
            )
            axes_accuracy.set_ylabel('accuracy', color='red')

            fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=axes_loss.transAxes)

        # psnr
        if self.__psnrs:
            axes_psnr = axes_loss.twinx()
            axes_psnr.plot(
                range(len(self.__psnrs)), self.__psnrs,
                label='psnr',
                color='red'
            )
            axes_psnr.set_ylabel(f'psnr({self.__brdf_max:.4})', color='red')

            fig.legend(loc='center right', bbox_to_anchor=(1, 0.5), bbox_transform=axes_loss.transAxes)

        plt.savefig(filename)

        plt.draw()
        plt.show()

    @staticmethod
    def __cos(x, y):
        light_z = x[:, 0:1, -1:]
        cos = light_z.squeeze()
        return torch.einsum('i,ij->ij', cos, y)

    def __loss(self, x, y, need_backward=False):
        y_pre = self.__model(x)
        # loss = self.__model.loss_function(y_pre, y)
        loss = self.__model.loss_function(self.__cos(x, y_pre), self.__cos(x, y))
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
        使用训练集与验证集训练模型

        :param epochs: 训练遍数
        """
        for epoch in range(epochs):
            # 训练
            for x, y in self.__train_dataloader:
                self.__loss(x, y, True)

            # 验证
            losses, cnt = 0, 0
            with torch.no_grad():
                for x, y in self.__valid_dataloader:
                    losses += self.__loss(x, y) * len(y)
                    cnt += len(y)
                    # self.__brdf_max = max(self.__brdf_max, torch.max(y))
                    self.__brdf_max = max(self.__brdf_max, torch.max(self.__cos(x, y)))
            loss = losses / cnt
            self.__losses.append(loss)

            psnr = 10 * math.log10(self.__brdf_max * self.__brdf_max / loss)
            self.__psnrs.append(psnr)

            print(f'epoch {epoch} :\nloss : {loss:.4}\npsnr : {psnr:.4}({self.__brdf_max:.4})')

            print(self.__model)

            if self.__model.lr:
                self.__model.lr.step()

    def pre(self, x):
        """
        使用模型预测结果

        :param x: 模型输入
        :return: 预测结果
        """
        return self.__model(x).detach()

    def __str__(self):
        return f'train on {len(self.__train_dataloader.dataset)}({self.__train_dataloader.batch_size}) samples for 32 epochs\n' \
               f'test on {len(self.__valid_dataloader.dataset)}({self.__valid_dataloader.batch_size}) samples\n' \
               f'get loss: {self.__losses[-1]:.4} psnr: {self.__psnrs[-1]:.4}({self.__brdf_max:.4})\n'
