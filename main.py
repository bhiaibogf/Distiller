from pkg import BsdfReader, Dataloader, Trainer
from pkg.model import *


def main():
    reader = BsdfReader('BSDF/ggx.txt', 4096, 1024)
    dataloader = Dataloader(reader, 64)
    model = GgxModel()
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(16)
    print(model)


if __name__ == '__main__':
    main()
