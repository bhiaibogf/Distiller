import torch
import torch.nn as nn


class PolynomialModel(nn.Module):
    def __init__(self, dim):
        super(PolynomialModel, self).__init__()
        self.__dim = dim
        self.__w = nn.Parameter(torch.rand(dim))
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        inputs = torch.ones(self.__dim, x.shape[0])
        for i in range(1, self.__dim):
            inputs[i] = inputs[i - 1] * x
        return torch.matmul(self.__w, inputs)
