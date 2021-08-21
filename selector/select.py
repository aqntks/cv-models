
import torch.nn as nn

from selector.cls_train import train, evaluate

from classification.mlp import MLP
from classification.cnn import CNN
from classification.resnet import ResNet
from classification.mobilenetv2 import MobileNetV2
from classification.mobilenetv3 import MobileNetV3

from optimizer.optimizer import optim


def classification(model_name, optimizer_name, DEVICE, EPOCHS, train_loader, test_loader):
    if model_name == 'mlp':
        model = MLP().to(DEVICE)
    if model_name == 'cnn':
        model = CNN().to(DEVICE)
    if model_name == 'resnet':
        model = ResNet().to(DEVICE)
    if model_name == 'mobilenetv2':
        model = MobileNetV2().to(DEVICE)
    if model_name == 'mobilenetv3s':
        model = MobileNetV3('mobilenet_v3_small').to(DEVICE)
    if model_name == 'mobilenetv3l':
        model = MobileNetV3('mobilenet_v3_large').to(DEVICE)

    optimizer = optim(optimizer_name, model)
    criterion = nn.CrossEntropyLoss()

    print(model)

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, criterion, DEVICE, epoch, log_interval=200)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, DEVICE)
        print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
            epoch, test_loss, test_accuracy))



