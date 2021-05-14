import torch
import torch.nn as nn
import torch.nn.functional as f


class MicrofacetBase(nn.Module):
    def __init__(self):
        super(MicrofacetBase, self).__init__()
        self._base_color = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self._alpha = nn.Parameter(torch.tensor([0.25]))
        self._eta = nn.Parameter(torch.tensor([1.45]))

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

    def d(self, nh):
        return 0

    def _schlick(self, vh):
        f_0 = ((self._eta - 1) / (self._eta + 1)) ** 2
        return f_0 + (1 - f_0) * (1 - vh) ** 5

    def _cook_torrance(self, vh):
        c = vh
        g = self._eta ** 2 + c ** 2 - 1
        if g > 0:
            g = torch.sqrt(g)
            a = (g - c) / (g + c)
            b = (c * (g + c) - 1) / (c * (g - c) + 1)
            return 0.5 * a ** 2 * (1 + b ** 2)
        return 1

    def f(self, vh):
        return self._cook_torrance(vh)

    def _c(self, nv):
        return nv / self._alpha / torch.sqrt(1 - nv ** 2)

    def g(self, light, normal, view):
        return normal.dot(light) * normal.dot(view)

    def forward(self, inputs):
        data_size = len(inputs)
        ls = torch.empty(data_size, 3)
        for i in range(data_size):
            light = inputs[i][0]
            normal = inputs[i][1]
            view = inputs[i][2]
            half = f.normalize(light + view, p=2, dim=0)
            ls[i] = self._base_color
            ls[i] *= self.d(torch.dot(normal, half)) * self.g(light, normal, view) * self.f(torch.dot(view, half))
            ls[i] /= 4 * torch.dot(light, normal) * torch.dot(normal, view)
        return ls

    def clamp_(self):
        self._alpha.data.clamp_(0, 1)
        self._eta.data.clamp_(1, 5)

    def __str__(self):
        return 'basecolor = ({}, {}, {})\nalpha = {}, eta = {}'.format(
            *self._base_color.tolist(), self._alpha.data.item(), self._eta.data.item())
