from pkg.model import *
from pkg.utils import BsdfReader2, Dataloader, Trainer


def main():
    reader = BsdfReader2('BSDF/pvc.txt', 10000, 1024)
    dataloader = Dataloader(reader, 64)
    model = PrincipledBrdf()
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(32)
    print(model)


if __name__ == '__main__':
    main()
