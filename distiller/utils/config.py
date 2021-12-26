import torch

if torch.cuda.is_available():
    USE_CUDA = True
    print('use cuda\n')
else:
    USE_CUDA = False
    print('use cpu\n')

USE_VEC = True

SHOW_LOSS = False
SHOW_MODEL = False
SHOW_IMG = False

WRITE_FILE = True
IMG_DIR = './img'
PARAMS_DIR = './params'

MERL_DIR = './MERL'

TRAIN_DATA_SIZE = 8192
VALID_DATA_SIZE = 1024
BATCH_SIZE = 1024

EPOCH = 32

SET_SEED = True

# for test
# TRAIN_DATA_SIZE = 80
# VALID_DATA_SIZE = 20
# BATCH_SIZE = 4
