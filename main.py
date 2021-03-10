from pkg import BlinnPhongReader, Dataloader, BlinnPhongModel, Trainer


def main():
    reader = BlinnPhongReader()
    dataloader = Dataloader(reader, 64)
    model = BlinnPhongModel()
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(128)
    print(model)


if __name__ == '__main__':
    main()
