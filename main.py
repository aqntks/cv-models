import torch

from data.load_data import data
from utils.common import model_save

from selector.select import classification


def main():
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('PyTorch 버전:', torch.__version__, ' Device:', DEVICE)

    BATCH_SIZE = 32
    EPOCHS = 10

    train_dataset, test_dataset = data('CIFAR_10')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    for (X_train, y_train) in train_loader:
        print('X_train:', X_train.size(), 'type:', X_train.type())
        print('y_train:', y_train.size(), 'type:', y_train.type())
        break

    model = classification('resnet', 'adam', DEVICE, EPOCHS, train_loader, test_loader)

    model_save(model, state_dict=True)


if __name__ == '__main__':
    main()

