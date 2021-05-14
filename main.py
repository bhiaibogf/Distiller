from pkg.model import *
from pkg.utils import BsdfReader, Dataloader, Trainer


def main():
    reader = BsdfReader('BSDF/alum-bronze.txt', 4096, 1024)
    dataloader = Dataloader(reader, 64)
    model = GgxModel()
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(16)
    print(model)


if __name__ == '__main__':
    main()
