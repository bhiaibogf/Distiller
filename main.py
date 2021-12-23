from pkg.model import *
from pkg.utils import Dataloader, Trainer, ModelReader


def main():
    if torch.cuda.is_available():
        print('use cuda')
    else:
        const.USE_CUDA = False
        print('use cpu')

    source_model = PrincipledBrdf(0.05, 0.7438)
    # source_model = PhongModel()
    if const.USE_CUDA:
        source_model = source_model.cuda()
    reader = ModelReader(8192, 2048, source_model)
    # reader = BsdfReader2('BSDF/blue-metallic-paint.txt', 8192, 2048)
    dataloader = Dataloader(reader, 64)

    model = GgxModel()
    if const.USE_CUDA:
        model = model.cuda()

    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(128)
    with open('params/' + model.__class__.__name__ + '.txt', 'a') as file:
        file.write(model.__str__() + '\n')


if __name__ == '__main__':
    main()
