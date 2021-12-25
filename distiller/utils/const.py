import math

import torch

if torch.cuda.is_available():
    USE_CUDA = True
    print('use cuda')
else:
    USE_CUDA = False
    print('use cpu')

USE_VEC = True

PI = math.pi

ZERO = torch.zeros(1, device='cuda' if USE_CUDA else 'cpu')
ZEROS = torch.zeros(3, device='cuda' if USE_CUDA else 'cpu')

ONE = torch.ones(1, device='cuda' if USE_CUDA else 'cpu')
ONES = torch.ones(3, device='cuda' if USE_CUDA else 'cpu')

POINT_ONE = torch.tensor([.1], device='cuda' if USE_CUDA else 'cpu')
POINT_OO_ONE = torch.tensor([.001], device='cuda' if USE_CUDA else 'cpu')
POINT_O_FOUR = torch.tensor([.04], device='cuda' if USE_CUDA else 'cpu')
