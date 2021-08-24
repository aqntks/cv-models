# Computer Vision을 위한 딥러닝 모델 구현 with Pytorch
![issue badge](https://img.shields.io/github/license/aqntks/cv-models?&color=blue)
![issue badge](https://img.shields.io/badge/build-passing-brightgreen)
![issue badge](https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-%EC%A7%80%EC%9B%90-orange)
[![LinkedIn Badge](http://img.shields.io/badge/LinkedIn-@InpyoHong-0072b1?style=flat&logo=linkedin&link=https://www.linkedin.com/in/inpyo-hong-886781212/)](https://www.linkedin.com/in/inpyo-hong-886781212/)

명령어 한 줄로 간편하게 다양한 딥러닝 모델을 학습하세요     

#### - 요구 조건 (Requirements) -
[Python>=3.6.0](https://www.python.org/) and [PyTorch>=1.7](https://pytorch.org/)
## 설치 (Install)
    git clone https://github.com/aqntks/cv-models
    
    cd cv-models     
    
    pip install -r requirements.txt

##  모델 학습 (Model Train)
    python main.py --mode classification --model resnet --optim adam --data CIFAR_10 --batch 32 --epoch 10
- args
> --mode : <classification, detection, segmentation >　　　　　 # 모델 유형  
--model : < resnet, mobilenetv3s, vgg-13, mnasnet-0.5 etc > &nbsp;&nbsp;  # 학습 모델   
--optim : < adam, SGD, AdaGrad, RMSprop etc > 　　&nbsp;　　　 # 옵티마이저  
--data :  < CIFAR_10, SVHN, Places365, STL10 etc >　　　　  # 학습 데이터  
--batch 　　　　　　　　　　　　　　　　　　　　　　&nbsp;　　# 배치 사이즈  
--epoch 　　　　　　　　　　　　　　　　　　　　　　　　# 세대 수  

## 모델 (MODELS)

#### * 분류 모델 (Classification) *
|MODEL|arg|paper|지원|
|---|---|---|---|
|MLP (Multi Layer Perceptron)|mlp|-|(O)|
|CNN (Convolutional Neural Network)|cnn|-|(O)|
|AlexNet|alexnet (이미지 사이즈 >= 256) |[[arxiv]](https://arxiv.org/pdf/1404.5997.pdf)|(O)|
|GoogleNet|-|[[arxiv]](https://arxiv.org/abs/1409.4842)|(준비중)|
|VGG|vgg-11, vgg-11_bn, vgg-13, vgg-13_bn, vgg-16, vgg-16_bn, vgg-19, vgg-19_bn|[[arxiv]](https://arxiv.org/pdf/1409.1556.pdf)|(O)|
|Inception|-|[[arxiv]](https://arxiv.org/pdf/1512.00567.pdf)|(준비중)|
|ResNet (Residual Neural Network)|resnet|[[arxiv]](https://arxiv.org/pdf/1512.03385.pdf)|(O)|
|SqueezeNet|squeezenet1_0, squeezenet1_1|[[arxiv]](https://arxiv.org/pdf/1602.07360.pdf)|(O)|
|MobileNetV2|mobilenetv2|[[arxiv]](https://arxiv.org/pdf/1801.04381.pdf)|(O)|
|DenseNet|densenet-121, densenet-161, densenet-169, densenet-201|[[arxiv]](https://arxiv.org/pdf/1608.06993.pdf)|(O)|
|ShuffleNetV2|shufflenetv2-x0.5, shufflenetv2-x1.0, shufflenetv2-x1.5, shufflenetv2-x2.0|[[arxiv]](https://arxiv.org/pdf/1807.11164.pdf)|(O)|
|MnasNet|mnasnet-0.5, mnasnet-0.75, mnasnet-1.0, mnasnet-1.3|[[arxiv]](https://arxiv.org/pdf/1807.11626.pdf)|(O)|
|MobileNetV3|mobilenetv3s, mobilenetv3l|[[arxiv]](https://arxiv.org/pdf/1905.02244.pdf)|(O)|
|vit|-|-|(준비중)|


#### * 탐지 모델 (Detection) *
1. yolov4 - (예정)
2. yolov5 - (예정)
3. Faster RCNN - (예정)
4. Mask R-CNN - (예정)
5. SSD - (예정)
#### * 분할 모델 (Segmentation) *
1. U-Net - (예정)
2. FCN - (예정)
5. DeepLabV3 - (예정)
6. DeepLabV3+ - (예정)
7. ReSeg - (예정)
#### * 생산적 적대 신경망 (GAN) *
