# Computer Vision을 위한 딥러닝 모델 구현 with Pytorch
![issue badge](https://img.shields.io/github/license/aqntks/cv-models?&color=blue)
![issue badge](https://img.shields.io/badge/build-passing-brightgreen)
![issue badge](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-%EC%A7%80%EC%9B%90-orange)
[![LinkedIn Badge](http://img.shields.io/badge/LinkedIn-@InpyoHong-0072b1?style=flat&logo=linkedin&link=https://www.linkedin.com/in/inpyo-hong-886781212/)](https://www.linkedin.com/in/inpyo-hong-886781212/)

명령어 한 줄로 간편하게 딥러닝 모델을 학습하세요  
손쉽게 다양한 모델들을 학습해볼 수 있습니다

모델의 미세 조정을 원하시면 [하이퍼 파라미터 튜닝](#하이퍼-파라미터-튜닝-model-tuning) 단계로 이동하세요

#### - 요구 조건 (Requirements) -
[Python>=3.6.0](https://www.python.org/) and [PyTorch>=1.7](https://pytorch.org/)
## 설치 (Install)
    git clone https://github.com/aqntks/cv-models
    
    cd cv-models     
    
    pip install -r requirements.txt

##  모델 학습 (Model Train)
    python main.py --mode classification --model resnet --optim adam --data CIFAR_10 --batch 32 --epoch 10
- args
> --mode : < classification, detection, segmentation >　　　　　# 모델 유형  
--model : < resnet, mobilenetv3s, vgg-13, mnasnet-0.5 etc > &nbsp;&nbsp;  # 학습 모델   
--optim : < adam, SGD, AdaGrad, RMSprop etc > 　　&nbsp;&nbsp;　　　 # 옵티마이저  
--data &nbsp;&nbsp;:  < CIFAR_10, SVHN, Places365, STL10 etc >　　　　　# 학습 데이터  
--batch 　　　　　　　　　　　　　　　　　　　　　　&nbsp;　　# 배치 사이즈  
--epoch 　　　　　　　　　　　　　　　　　　　　　　　　# 세대 수  
--img 　　　　　　　　　　　　　　　　　　　　　　　　　# 이미지 사이즈 (지정 안하면 기본 데이터 사이즈)
## 모델 (MODELS)

#### * 분류 모델 (Classification) *
|MODEL|arg|paper|지원|
|---|---|---|---|
|MLP (Multi Layer Perceptron)|mlp|<center>-</center>|<center>(O)</center>|
|CNN (Convolutional Neural Network)|cnn|[[paper]](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)|<center>(O)</center>|
|AlexNet|alexnet (이미지 사이즈 >= 256) |[[arxiv]](https://arxiv.org/pdf/1404.5997.pdf)|<center>(O)</center>|
|GoogleNet|<center>-</center>|[[arxiv]](https://arxiv.org/abs/1409.4842)|<center>(준비중)</center>|
|VGG|vgg-11, vgg-11_bn, vgg-13, vgg-13_bn, </br> vgg-16, vgg-16_bn, vgg-19, vgg-19_bn|[[arxiv]](https://arxiv.org/pdf/1409.1556.pdf)|<center>(O)</center>|
|Inception|<center>-</center>|[[arxiv]](https://arxiv.org/pdf/1512.00567.pdf)|<center>(준비중)</center>|
|ResNet (Residual Neural Network)|resnet|[[arxiv]](https://arxiv.org/pdf/1512.03385.pdf)|<center>(O)</center>|
|SqueezeNet|squeezenet1_0, squeezenet1_1|[[arxiv]](https://arxiv.org/pdf/1602.07360.pdf)|<center>(O)</center>|
|MobileNetV2|mobilenetv2|[[arxiv]](https://arxiv.org/pdf/1801.04381.pdf)|<center>(O)</center>|
|DenseNet|densenet-121, densenet-161, </br> densenet-169, densenet-201|[[arxiv]](https://arxiv.org/pdf/1608.06993.pdf)|<center>(O)</center>|
|ShuffleNetV2|shufflenetv2-x0.5, shufflenetv2-x1.0, </br> shufflenetv2-x1.5, shufflenetv2-x2.0|[[arxiv]](https://arxiv.org/pdf/1807.11164.pdf)|<center>(O)</center>|
|MnasNet|mnasnet-0.5, mnasnet-0.75, </br> mnasnet-1.0, mnasnet-1.3|[[arxiv]](https://arxiv.org/pdf/1807.11626.pdf)|<center>(O)</center>|
|MobileNetV3|mobilenetv3s, mobilenetv3l|[[arxiv]](https://arxiv.org/pdf/1905.02244.pdf)|<center>(O)</center>|
|VIT|vit (이미지 사이즈 = 224)|[[paper]](https://openreview.net/pdf?id=YicbFdNTTy)|<center>(O)</center>|

#### * 탐지 모델 (Detection) *
|MODEL|arg|paper|지원|
|---|---|---|---|
|Faster R-CNN|-|<center>-</center>|<center>(준비중)</center>|
|Mask R-CNN|-|<center>-</center>|<center>(준비중)</center>|
|SSD|[[github]](https://github.com/amdegroot/ssd.pytorch)|<center>-</center>|<center>(준비중)</center>|
|yolov4|[[arxiv]](https://arxiv.org/pdf/2004.10934.pdf)|<center>-</center>|<center>(준비중)</center>|
|yolov5|[[github]](https://github.com/ultralytics/yolov5)|<center>-</center>|<center>(준비중)</center>|

#### * 분할 모델 (Segmentation) *
1. U-Net - (예정)
2. FCN - (예정)
3. DeepLabV3 - (예정)
4. DeepLabV3+ - (예정)
5. ReSeg - (예정)

#### * 생산적 적대 신경망 (GAN) *

##  예측 결과 (Prediction results)
data/label.txt에 학습한 모델의 클래스 명을 정의하세요   

    ##### data/label.txt #####

    airplane
    automobile
    bird
    cat
    deer
    dog
    frog
    horse
    ship
    truck

검출  
    
    python predict.py --model resnet --weight result/model.pt --img data/test.jpg


## 전이 학습 (Transfer Learning)
사전 학습된 모델이 있다면 --weights 인수에 사전 학습 모델을 추가해 주세요  
학습을 위해 선택한 모델과 같은 구조로 학습한 모델이어야 합니다.

    
    python main.py --weights pretrained_model.pth --densenet-121 --mode classification

## 하이퍼 파라미터 튜닝 (model tuning)
    
    python export.py

export.py를 통해 학습 할 모델의 하이퍼 파라미터 튜닝을 진행하세요

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='classification')
    parser.add_argument('--model', type=str, default='densenet-169')
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