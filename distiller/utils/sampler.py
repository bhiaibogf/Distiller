import math

import torch
import torch.nn.functional as f


class Sampler:
    @staticmethod
    def disk(n):
        xi_1 = torch.rand(n)
        xi_2 = torch.rand(n)

        r = torch.sqrt(xi_1)
        theta = 2 * math.pi * xi_2

        x, y = r * torch.cos(theta), r * torch.sin(theta)
        return torch.stack((x, y), 1)

    @staticmethod
    def sphere(n):
        xi_1 = torch.rand(n)
        xi_2 = torch.rand(n)

        cos_theta = 1 - 2 * xi_1
        sin_theta = torch.sqrt(1 - cos_theta * cos_theta)

        phi = 2 * math.pi * xi_2
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        x, y, z = sin_theta * cos_phi, sin_theta * sin_phi, cos_theta
        return torch.stack((x, y, z), 1)

    @staticmethod
    def sphere_using_randn(n):
        x = torch.randn(n)
        y = torch.randn(n)
        z = torch.randn(n)
        vec = torch.stack((x, y, z), 1)
        return f.normalize(vec, p=2, dim=1)

    @staticmethod
    def hemisphere(n):
        xi_1 = torch.rand(n)
        xi_2 = torch.rand(n)

        cos_theta = 1 - xi_1
        sin_theta = torch.sqrt(1 - cos_theta * cos_theta)

        phi = 2 * math.pi * xi_2
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        x, y, z = sin_theta * cos_phi, sin_theta * sin_phi, cos_theta
        return torch.stack((x, y, z), 1)

    @staticmethod
    def cos_hemisphere(n):
        xi_1 = torch.rand(n)
        xi_2 = torch.rand(n)

        sin_theta = torch.sqrt(xi_1)
        cos_theta = torch.sqrt(1 - xi_1)

        phi = 2 * math.pi * xi_2
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        x, y, z = sin_theta * cos_phi, sin_theta * sin_phi, cos_theta
        return torch.stack((x, y, z), 1)
