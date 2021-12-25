import torch

if torch.cuda.is_available():
    USE_CUDA = True
    print('use cuda')
else:
    USE_CUDA = False
    print('use cpu')

USE_VEC = True

SHOW_LOSS = True
SHOW_MODEL = True
SHOW_IMG = True

WRITE_FILE = False
