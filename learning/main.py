import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MyModel(nn.Module):
    def __init__(self, dim):
        super(MyModel, self).__init__()
        self.__dim = dim
        self.__w = nn.Parameter(torch.randn(dim))
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        inputs = torch.ones(self.__dim, x.shape[0])
        for i in range(1, self.__dim):
            inputs[i] = inputs[i - 1] * x
        return torch.matmul(self.__w, inputs)


def main():
    xs = torch.linspace(-3, 3, 1024)
    ys = torch.sin(xs)

    dataset = TensorDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MyModel(5)
    for epoch in range(100):
        cnt = 0
        for xx, yy in dataloader:
            cnt += 1
            yy_pre = model(xx)
            # print(xx, var(torch.tensor(yy)), yy_pre)
            loss = model.loss_function(yy_pre, yy)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            if cnt % 3 == 0:
                print('loss =', loss)

    yy_pre = [model(torch.tensor([x])) for x in xs]
    plt.title('result')
    plt.plot(xs, ys)
    plt.plot(xs, yy_pre)
    plt.show()


if __name__ == '__main__':
    main()
