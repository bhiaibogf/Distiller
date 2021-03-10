import matplotlib.pyplot as plt

from pkg import PolynomialReader, Dataloader, PolynomialModel, Trainer


def main():
    reader = PolynomialReader()
    dataloader = Dataloader(reader, 64)
    model = PolynomialModel(5)
    trainer = Trainer(model, dataloader.get_train_dataloader(), dataloader.get_valid_dataloader())
    trainer.train(128)

    x, y = dataloader.get_data()
    y_pre = trainer.pre(x)

    plt.title('result')
    plt.plot(x, y)
    plt.plot(x, y_pre)
    plt.show()


if __name__ == '__main__':
    main()
