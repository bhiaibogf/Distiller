import torch
import torch.nn as nn


class BrdfBase(nn.Module):
    def __init__(self):
        super(BrdfBase, self).__init__()

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def __str__(self):
        for name, param in self.named_parameters():
            name = name[name.rfind('__') + 2:]
            if param.size() == (1,):
                print("{} = {}".format(name, param.data.item()))
            else:
                print("{} = {}".format(name, param.data.tolist()))
        return ''

    def _eval(self, light, normal, view):
        return torch.ones(3)

    def forward(self, inputs):
        data_size = len(inputs)
        ls = torch.empty(data_size, 3)
        for i in range(data_size):
            light = inputs[i][0]
            normal = torch.tensor([0.0, 0.0, 1.0])
            view = inputs[i][1]
            ls[i] = self._eval(light, normal, view)
        return ls
