from pkg import BsdfReader, Dataloader, Trainer, BlinnPhongModel


def main():
    reader = BsdfReader('BSDF/bsdf.txt', 4096, 1024)
    dataloader = Dataloader(reader, 64)
    model = BlinnPhongModel()
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(32)
    print(model)


if __name__ == '__main__':
    main()
