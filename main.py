import os

from distiller.model import *
from distiller.utils import Dataloader, Trainer, ModelReader, const


def main():
    if torch.cuda.is_available():
        print('use cuda')
    else:
        const.USE_CUDA = False
        print('use cpu')

    # source_model = PrincipledBrdf(0.05, 0.7438)
    source_model = PrincipledBrdf(0.8, 0.2)
    # source_model = PhongModel()
    if const.USE_CUDA:
        source_model = source_model.cuda()
    reader = ModelReader(8192, 2048, source_model)
    # reader = BsdfReader2('BSDF/blue-metallic-paint.txt', 8192, 2048)
    dataloader = Dataloader(reader, 64)

    # for test
    # reader = ModelReader(80, 20, source_model)
    # dataloader = Dataloader(reader, 4)

    model = GgxModel()
    if const.USE_CUDA:
        model = model.cuda()

    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(32)

    source_model_name = source_model.__class__.__name__.split('Model')[0]
    target_model_name = model.__class__.__name__.split('Model')[0]

    pic_dir = './img'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    cnt = 0
    while os.path.exists(f'{pic_dir}/{source_model_name}_{target_model_name}_{cnt}.png'):
        cnt += 1
    trainer.plot(f'{pic_dir}/{source_model_name}_{target_model_name}_{cnt}.png')

    params_dir = './params'
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    with open(f'{params_dir}/{source_model_name}_{target_model_name}.txt', 'a') as file:
        file.write(f'{cnt}:\n')
        file.write(f'source:\n{source_model.__str__()}\n')
        file.write(f'target:\n{model.__str__()}\n\n')


if __name__ == '__main__':
    main()
