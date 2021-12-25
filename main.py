import os

from distiller.model import *
from distiller.utils import Dataloader, Trainer, ModelReader, config, timer


def sample():
    # source_model = PrincipledBrdf(0.05, 0.7438)
    # source_model = PrincipledBrdf(0.8, 0.2)
    # source_model = PhongModel()
    # if config.USE_CUDA:
    #     source_model = source_model.cuda()
    # reader = ModelReader(config.TRAIN_DATA_SIZE, config.VALID_DATA_SIZE, source_model)

    source_model = MerlModel('blue-metallic-paint')
    reader = ModelReader(config.TRAIN_DATA_SIZE, config.VALID_DATA_SIZE, source_model)
    # reader = BsdfReader2(f'BSDF/{source_model}.txt', TRAIN_DATA_SIZE, VALID_DATA_SIZE)

    dataloader = Dataloader(reader, config.BATCH_SIZE)

    return source_model, dataloader


def train(dataloader):
    target_model = GgxModel()
    if config.USE_CUDA:
        target_model = target_model.cuda()

    trainer = Trainer(target_model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(config.EPOCH)

    return target_model, trainer


def output(source_model, target_model, trainer):
    source_model_name = source_model.__class__.__name__.split('Model')[0]
    target_model_name = target_model.__class__.__name__.split('Model')[0]

    img_dir = config.IMG_DIR
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    cnt = 0
    while os.path.exists(f'{img_dir}/{source_model_name}_{target_model_name}_{cnt}.png'):
        cnt += 1
    img_file = f'{img_dir}/{source_model_name}_{target_model_name}_{cnt}.png'
    trainer.plot(img_file)
    print('drawn to ' + img_file)

    params_dir = config.PARAMS_DIR
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    params_file = f'{params_dir}/{source_model_name}_{target_model_name}.txt'
    with open(params_file, 'a') as file:
        file.write(f'{cnt}:\n')
        file.write(f'source:\n{source_model.__str__()}\n')
        file.write(f'target:\n{target_model.__str__()}\n')
        file.write(f'{trainer}\n\n')
    print('written to ' + params_file)


def main():
    print('sampling...')
    source_model, dataloader = sample()
    timer.update_time('sampling')

    print('training...')
    target_model, trainer = train(dataloader)
    timer.update_time('training')

    print('output...')
    print(target_model)
    print(trainer)
    if config.WRITE_FILE:
        output(source_model, target_model, trainer)
    elif config.SHOW_IMG:
        trainer.plot(None)
    timer.update_time('output')


if __name__ == '__main__':
    main()
