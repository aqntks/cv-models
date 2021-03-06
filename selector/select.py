import torch.nn as nn
import torch
from selector.cls_train import train, evaluate

from classification.mlp import MLP
from classification.cnn import CNN
from classification.resnet import ResNet
from classification.mobilenetv2 import MobileNetV2
from classification.mobilenetv3 import MobileNetV3
from classification.alexnet import AlexNet
# from classification.googlenet import GoogLeNet
from classification.squeezenet import SqueezeNet
from classification.vgg import makeVgg
from classification.mnasnet import MNASNet
from classification.shufflenetv2 import ShuffleNetV2
from classification.densenet import DenseNet
# from classification.inception import Inception3

from optimizer.optimizer import optim

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from transformer.linformer import Linformer
from classification.vit.efficient import ViT


def vit_train(DEVICE, img_size, class_count):
    # Linformer
    efficient_transformer = Linformer(
        dim=128,
        seq_len=49 + 1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64
    )

    # vit
    model = ViT(
        dim=128,
        image_size=img_size,
        patch_size=32,
        num_classes=class_count,
        transformer=efficient_transformer,
        channels=3,
    ).to(DEVICE)

    return model


def get_classification(model_name, DEVICE, img_size, class_count, dropout, memoryEF):
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
    if model_name == 'alexnet':
        model = AlexNet().to(DEVICE)
    if model_name == 'googlenet':      # ?????? ?????????
        model = GoogLeNet().to(DEVICE)
    if model_name == 'squeezenet1_0':
        model = SqueezeNet('1_0').to(DEVICE)
    if model_name == 'squeezenet1_1':
        model = SqueezeNet('1_1').to(DEVICE)
    if model_name.split('-')[0] == 'vgg':
        model_name = model_name.split('-')[0] + model_name.split('-')[1]
        model = makeVgg(model_name).to(DEVICE)
    if model_name.split('-')[0] == 'mnasnet':
        model = MNASNet(float(model_name.split('-')[1]), dropout=dropout).to(DEVICE)
    if model_name.split('-')[0] == 'shufflenetv2':
        version = model_name.split('-')[1]
        if version == 'x0.5':
            channels = [24, 48, 96, 192, 1024]
        if version == 'x1.0':
            channels = [24, 116, 232, 464, 1024]
        if version == 'x1.5':
            channels = [24, 176, 352, 704, 1024]
        if version == 'x2.0':
            channels = [24, 244, 488, 976, 2048]
        model = ShuffleNetV2(channels).to(DEVICE)
    if model_name.split('-')[0] == 'densenet':
        version = model_name.split('-')[1]
        if version == '121':
            model = DenseNet(32, (6, 12, 24, 16), 64, drop_rate=dropout, memory_efficient=memoryEF).to(DEVICE)
        if version == '161':
            model = DenseNet(48, (6, 12, 36, 24), 96, drop_rate=dropout, memory_efficient=memoryEF).to(DEVICE)
        if version == '169':
            model = DenseNet(32, (6, 12, 32, 32), 64, drop_rate=dropout, memory_efficient=memoryEF).to(DEVICE)
        if version == '201':
            model = DenseNet(32, (6, 12, 48, 32), 64, drop_rate=dropout, memory_efficient=memoryEF).to(DEVICE)
    if model_name == 'vit':
        model = vit_train(DEVICE, img_size, class_count)
    # if model_name == 'inception':
    #     model = Inception3().to(DEVICE)

    return model


def classification(model_name, optimizer_name, DEVICE, EPOCHS, train_loader, test_loader, img_size, class_count, opt):
    weights = opt.weights
    lr, momentum, dropout, memoryEF = opt.lr, opt.momentum, opt.dropout, opt.memoryEF
    model = get_classification(model_name, DEVICE, img_size, class_count, dropout, memoryEF)

    optimizer = optim(optimizer_name, model, lr, momentum)
    criterion = nn.CrossEntropyLoss()

    if weights != '':
        model.load_state_dict(torch.load(weights))

    print(model)

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, criterion, DEVICE, epoch, log_interval=200)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, DEVICE)
        print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
            epoch, test_loss, test_accuracy))

    return model


def detection():
    print("???????????? ???????????????...")


def segmentation():
    print("???????????? ???????????????...")
