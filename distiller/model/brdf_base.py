import threading

import torch
import torch.nn as nn

from distiller.utils import config, funcs


class BrdfBase(nn.Module):
    def __init__(self):
        super(BrdfBase, self).__init__()

        self.loss_function = nn.MSELoss()
        self.optimizer = None
        self.lr = None

    def __str__(self):
        output = ''
        for name, param in self.named_parameters():
            class_name = self.__class__.__name__
            if name.find(class_name) != -1:
                name = name[len(class_name) + 1:]
            while name[0] == '_':
                name = name[1:]
            if param.size() == (1,):
                output += "{} = {:.4}\n".format(name, param.data.item())
            else:
                output += "{} = [{:.4}, {:.4}, {:.4}] #{}\n".format(name, *param.data.tolist(),
                                                                    funcs.to_hex(param.data.tolist()))
        return output

    def _shade(self, light, normal, view):
        pass

    def _eval(self, light, normal, view):
        pass

    def _handler(self, x, y, i):
        light = x[0]
        normal = torch.tensor([0.0, 0.0, 1.0])
        if config.USE_CUDA:
            normal = normal.cuda()
        view = x[1]
        y[i] = self._eval(light, normal, view)

    def forward(self, inputs):
        if config.USE_VEC:
            light = inputs[:, 0:1, :].squeeze()
            normal = torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0)
            if config.USE_CUDA:
                normal = normal.cuda()
            view = inputs[:, 1:, :].squeeze()
            return self._shade(light, normal, view)
        else:
            data_size = len(inputs)
            result = torch.empty(data_size, 3)
            if config.USE_CUDA:
                result = result.cuda()
                for i in range(data_size):
                    self._handler(inputs[i], result, i)
            else:
                threads = []
                for i in range(data_size):
                    threads.append(threading.Thread(target=self._handler, args=(inputs[i], result, i)))
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            return result
