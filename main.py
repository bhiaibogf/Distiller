from pkg.model import *
from pkg.utils import BsdfReader2, Dataloader, Trainer


def main():
    reader = BsdfReader2('BSDF/ggx.txt', 4096, 1024)
    dataloader = Dataloader(reader, 64)
    model = BeckmannModel()
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(32)
    print(model)


if __name__ == '__main__':
    main()
