import os
from multiprocessing import cpu_count

import torch

from pkg.model import *
from pkg.utils import BsdfReader2, Dataloader, Trainer


def main():
    cpu_num = cpu_count()  # 自动获取最大核心数目
    print(cpu_num)
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    torch.set_num_interop_threads(cpu_num)
    print(torch.get_num_threads())
    print(torch.get_num_interop_threads())

    reader = BsdfReader2('BSDF/ggx.txt', 4096, 1024)
    dataloader = Dataloader(reader, 64)
    model = BeckmannModel()
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(32)
    print(model)


if __name__ == '__main__':
    main()
