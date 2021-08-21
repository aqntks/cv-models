import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms, datasets

from classification.resnet import ResNet
from train import train, evaluate


def data():
    train_dataset = datasets.CIFAR10(root="data/CIFAR_10",
                                     train=True,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    test_dataset = datasets.CIFAR10(root="data/CIFAR_10",
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    return train_dataset, test_dataset


def main():
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('PyTorch 버전:', torch.__version__, ' Device:', DEVICE)

    BATCH_SIZE = 32
    EPOCHS = 10

    train_dataset, test_dataset = data()

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

    pltsize = 1
    plt.figure(figsize=(10 * pltsize, pltsize))

    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.axis('off')
        plt.imshow(np.transpose(X_train[i], (1, 2, 0)))
        plt.title('Class: ' + str(y_train[i].item()))

    model = ResNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(model)

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, criterion, DEVICE, epoch, log_interval=200)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, DEVICE)
        print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
            epoch, test_loss, test_accuracy))


if __name__ == '__main__':
    main()

