import torch


class Trainer:

    def __init__(self, model, train_dataloader, valid_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

    def __loss(self, x, y, need_backward=False):
        y_pre = self.model(x)
        loss = self.model.loss_function(y_pre, y)
        if need_backward:
            loss.backward()
            self.model.optimizer.step()
            self.model.optimizer.zero_grad()
        return loss

    def train(self, epochs):
        """
        使用训练集与测试机训练模型
        :param epochs: 训练遍数
        :return: None
        """
        for epoch in range(epochs):
            # 训练集
            self.model.train()
            for x, y in self.train_dataloader:
                self.__loss(x, y, True)

            # 测试集
            self.model.eval()
            with torch.no_grad():
                losses, cnt = 0, 0
                for x, y in self.valid_dataloader:
                    losses += self.__loss(x, y) * len(y)
                    cnt += len(y)
            loss = losses / cnt
            print(epoch, loss)

    def pre(self, x):
        """
        使用模型预测结果
        :param x: 模型输入
        :return: 预测结果
        """
        return self.model(x).detach()
