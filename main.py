import sys

import torch
import argparse

from data.load_data import data
from utils.common import model_save, lightWeight

from selector.select import classification, detection, segmentation

model_list = ['mlp', 'cnn', 'resnet', 'alexnet', 'mobilenetv2', 'mobilenetv3s', 'mobilenetv3l', 'alexnet', 'squeezenet1_0', 'squeezenet1_1']


def main(opt):
    mode, model_name, optimizer, img_data, batch_size, epochs, img_size, light_weight \
        = opt.mode, opt.model, opt.optim, opt.data, opt.batch, opt.epoch, opt.img, opt.light_weight

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('PyTorch 버전:', torch.__version__, ' Device:', device)

    try:
        train_dataset, test_dataset = data(img_data, img_size)
        class_count = len(train_dataset.classes)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        for (X_train, y_train) in train_loader:
            print('X_train:', X_train.size(), 'type:', X_train.type())
            print('y_train:', y_train.size(), 'type:', y_train.type())
            break
    except:
        print('지원하지 않는 데이터입니다')
        sys.exit()
    model = classification(model_name, optimizer, device, epochs, train_loader, test_loader, img_size, class_count, opt)
    try:
        if mode == 'classification':
            model = classification(model_name, optimizer, device, epochs, train_loader, test_loader, img_size, class_count, opt)
        elif mode == 'detection':
            detection()
        elif mode == 'segmentation':
            segmentation()

        if light_weight:
            model = lightWeight(model)
        model_save(model, state_dict=True)
    except:
        print('지원하지 않는 모델입니다')
        sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='classification')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--data', type=str, default='CIFAR_10')
    parser.add_argument('--batch', type=int, default=32, help='batch size. 2의 배수를 추천합니다.')
    parser.add_argument('--epoch', type=int, default=10, help='학습 세대 수')
    parser.add_argument('--weights', type=str, default='', help='전이학습을 이용하려면 학습을 원하는 모델 구조로 사전 학습된 모델을 넣어주세요')
    parser.add_argument('--img', type=int, default=-1, help='이미지 사이즈입니다. -1을 지정할 경우 기본 이미지 사이즈로 데이터를 다운받습니다.')
    parser.add_argument('--lr', type=float, default='0.001', help='learning rate 학습률 수치입니다')
    parser.add_argument('--momentum', type=float, default='0.937', help='SGD optimizer를 사용하는 경우 모멘텀 값을 설정하세요')
    parser.add_argument('--dropout', type=float, default='0.2', help='MNASNet, DenseNet 지원. 레이어의 dropout 비율을 적용하세요')
    parser.add_argument('--memoryEF', type=bool, default=False, help='DenseNet 지원. True를 설정하면 효율적인 메모리 학습이 가능합니다. 속도는 느려집니다')
    parser.add_argument('--light_weight', type=bool, default=False, help='True로 설정하면 학습을 마친 후 모델 경량화 작업을 진행합니다')
    opt = parser.parse_args()
    main(opt)
