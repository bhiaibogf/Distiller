import matplotlib.pyplot as plt

from pkg import PolynomialModel, Trainer, Reader


def main():
    reader = Reader(64)
    polynomial_model = PolynomialModel(5)
    trainer = Trainer(polynomial_model, reader.get_train_dataloader(), reader.get_valid_dataloader())
    trainer.train(128)

    x, y = reader.get_data()
    y_pre = trainer.pre(x)

    plt.title('result')
    plt.plot(x, y)
    plt.plot(x, y_pre)
    plt.show()


if __name__ == '__main__':
    main()
