import argparse
from main import main


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
    opt = parser.parse_args()
    main(opt)
