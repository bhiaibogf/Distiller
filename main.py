from pkg.model import *
from pkg.utils import BsdfReader2, Dataloader, Trainer


def main():
    reader = BsdfReader2('BSDF/blue-metallic-paint.txt', 8192, 2048)
    dataloader = Dataloader(reader, 64)
    model = PhongModel()
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(128)
    with open('params/' + model.__class__.__name__ + '.txt', 'a') as file:
        file.write(model.__str__() + '\n')


if __name__ == '__main__':
    main()
