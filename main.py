from pkg.model import *
from pkg.utils import BsdfReader2, Dataloader, Trainer


def main():
    reader = BsdfReader2('BSDF/blue-metallic-paint.txt', 4096, 2048)
    dataloader = Dataloader(reader, 64)
    model = PhongModel()
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(128)
    print(model)


if __name__ == '__main__':
    main()
