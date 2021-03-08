import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as Var


class MyModel(nn.Module):
    def __init__(self, dim):
        super(MyModel, self).__init__()
        self.__dim = dim
        self.__w = nn.Parameter(torch.randn(dim, dtype=torch.float64))
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        inputs = torch.tensor([x ** i for i in range(self.__dim)], dtype=torch.float64)
        return torch.matmul(self.__w, inputs)


def main():
    xs = np.arange(-3, 3, 0.05)
    ys = np.sin(xs)

    model = MyModel(5)
    for epoch in range(600):
        loss = Var(torch.zeros(1))
        for xx, yy in zip(xs, ys):
            yy_pre = model(xx)
            # print(xx, var(torch.tensor(yy)), yy_pre)
            loss = loss + model.loss_function(yy_pre, Var(torch.tensor(yy)))
        if epoch % 200 == 0:
            print('loss =', loss)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    yy_pre = [model(x) for x in xs]
    plt.title('result')
    plt.plot(xs, ys)
    plt.plot(xs, yy_pre)
    plt.show()


if __name__ == '__main__':
    main()
