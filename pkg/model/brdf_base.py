import threading

import torch
import torch.nn as nn

from pkg.model.utils import to_hex


class BrdfBase(nn.Module):
    def __init__(self):
        super(BrdfBase, self).__init__()

        self.loss_function = nn.MSELoss()
        self.optimizer = None
        self.lr = None

    def __str__(self):
        for name, param in self.named_parameters():
            class_name = self.__class__.__name__
            if name.find(class_name) != -1:
                name = name[len(class_name) + 1:]
            while name[0] == '_':
                name = name[1:]
            if param.size() == (1,):
                print("{} = {}".format(name, param.data.item()))
            else:
                print("{} = {} #{}".format(name, param.data.tolist(), to_hex(param.data.tolist())))
        return ''

    def _eval(self, light, normal, view):
        return torch.ones(3)

    def _handler(self, x, y, i):
        light = x[0]
        normal = torch.tensor([0.0, 0.0, 1.0])
        view = x[1]
        y[i] = self._eval(light, normal, view)

    def forward(self, inputs):
        data_size = len(inputs)
        result = torch.empty(data_size, 3)
        threads = []
        for i in range(data_size):
            threads.append(threading.Thread(target=self._handler, args=(
                inputs[i], result, i)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
