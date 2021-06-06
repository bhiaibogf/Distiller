import torch
import torch.nn as nn
import torch.nn.functional as f

from pkg.model.brdf_base import BrdfBase
from pkg.model.utils import *


class PrincipledBrdf(BrdfBase):
    def __init__(self):
        super(PrincipledBrdf, self).__init__()
        self.__base_color = nn.Parameter(torch.tensor([0.8, 0.8, 0.8]))

        self.__metallic = nn.Parameter(torch.tensor([0.0]))
        self.__subsurface = nn.Parameter(torch.tensor([0.0]))
        self.__specular = nn.Parameter(torch.tensor([0.5]))
        self.__roughness = nn.Parameter(torch.tensor([0.5]))
        self.__specular_tint = nn.Parameter(torch.tensor([0.0]))
        self.__anisotropic = nn.Parameter(torch.tensor([0.0]))
        self.__sheen = nn.Parameter(torch.tensor([0.0]))
        self.__sheen_tint = nn.Parameter(torch.tensor([0.5]))
        self.__clear_coat = nn.Parameter(torch.tensor([0.0]))
        self.__clear_coat_gloss = nn.Parameter(torch.tensor([1.0]))

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def clamp_(self):
        for param in self.parameters():
            param.data.clamp_(0, 1)

    @staticmethod
    def _schlick(u):
        m = torch.clamp(1 - u, 0, 1)
        return quick_pow(m, 5)

    @staticmethod
    def _d_gtr1(cos_nm, alpha):
        if alpha >= 1:
            return 1 / PI
        a2 = sqr(alpha)
        t = 1 + (a2 - 1) * sqr(cos_nm)
        return (a2 - 1) / (PI * torch.log(a2) * t)

    @staticmethod
    def _d_gtr2(cos_nm, alpha):
        a2 = sqr(alpha)
        t = 1 + (a2 - 1) * sqr(cos_nm)
        return a2 / (PI * sqr(t))

    @staticmethod
    def _d_gtr2_aniso(cos_nm, cos_xm, cos_ym, ax, ay):
        return 1 / (PI * ax * ay * sqr(
            sqr(cos_xm / ax) + sqr(cos_ym / ay) + sqr(cos_nm)
        ))

    @staticmethod
    def _g(cos_ns, alpha):
        a = sqr(alpha)
        b = sqr(cos_ns)
        return 1 / (cos_ns + torch.sqrt(a + b - a * b))

    @staticmethod
    def _g_aniso(cos_ns, cos_xs, cos_ys, ax, ay):
        return 1 / (cos_ns + torch.sqrt(
            sqr(cos_xs * ax) + sqr(cos_ys * ay) + sqr(cos_ns)
        ))

    @staticmethod
    def _aniso(base, x, y):
        return base.dot(x), base.dot(y)

    def _eval(self, light, normal, view):
        x = torch.tensor([1.0, 0.0, 0.0])
        y = torch.tensor([0.0, 1.0, 0.0])
        half = f.normalize(light + view, p=2, dim=0)

        cos_nl = normal.dot(light)
        cos_nv = normal.dot(view)
        cos_nh = normal.dot(half)
        cos_hl = half.dot(light)

        if cos_nl < 0 or cos_nv < 0:
            return torch.zeros(3)

        cd_lin = mon2lin(self.__base_color)
        cd_lum = .3 * cd_lin[0] + .6 * cd_lin[1] + .1 * cd_lin[2]

        if cd_lum > 0:
            c_tint = cd_lin / cd_lum
        else:
            c_tint = torch.zeros(3)
        c_spec0 = torch.lerp(
            self.__specular * .08 * torch.lerp(
                torch.ones(3),
                c_tint,
                self.__specular_tint
            ),
            cd_lin,
            self.__metallic
        )
        c_sheen = torch.lerp(torch.ones(3), c_tint, self.__sheen_tint)

        fresnel_l, fresnel_v = self._schlick(cos_nl), self._schlick(cos_nv)
        fresnel_diffuse_90 = 0.5 + 2 * quick_pow(cos_hl, 2) * self.__roughness
        fresnel_diffuse = torch.lerp(torch.ones(1), fresnel_diffuse_90, fresnel_l) * \
                          torch.lerp(torch.ones(1), fresnel_diffuse_90, fresnel_v)

        fresnel_ss_90 = sqr(cos_hl) * self.__roughness
        fresnel_ss = torch.lerp(torch.ones(1), fresnel_ss_90, fresnel_l) * \
                     torch.lerp(torch.ones(1), fresnel_ss_90, fresnel_v)
        ss = 1.25 * (fresnel_ss * (1 / (cos_nl + cos_nv) - .5) + .5)

        # specular
        aspect = torch.sqrt(1 - self.__anisotropic * .9)
        ax = max(.001, quick_pow(self.__roughness, 2) / aspect)
        ay = max(.001, quick_pow(self.__roughness, 2) * aspect)
        Ds = self._d_gtr2_aniso(cos_nh, *self._aniso(half, x, y), ax, ay)
        FH = self._schlick(cos_hl)
        Fs = torch.lerp(c_spec0, torch.ones(3), FH)
        Gs = self._g_aniso(cos_nl, *self._aniso(light, x, y), ax, ay) * \
             self._g_aniso(cos_nv, *self._aniso(view, x, y), ax, ay)
        specular = Gs * Fs * Ds

        Fsheen = FH * self.__sheen * c_sheen
        diffuse = ((1 / PI)
                   * torch.lerp(fresnel_diffuse, ss, self.__subsurface)
                   * cd_lin
                   + Fsheen
                   ) * (1 - self.__metallic)

        # clear_coat
        Dr = self._d_gtr1(cos_nh, torch.lerp(torch.tensor([.1]), torch.tensor([.001]), self.__clear_coat_gloss))
        Fr = torch.lerp(torch.tensor(.04), torch.tensor(1.0), FH)
        Gr = self._g(cos_nl, .25) * self._g(cos_nv, .25)
        clear_coat = .25 * self.__clear_coat * Gr * Fr * Dr

        return diffuse + specular + clear_coat


if __name__ == '__main__':
    print(PrincipledBrdf())
