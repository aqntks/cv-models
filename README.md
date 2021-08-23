# Computer Vision을 위한 딥러닝 모델 구현 with Pytorch
#### - 요구 조건 -
[Python>=3.6.0](https://www.python.org/) and [PyTorch>=1.7](https://pytorch.org/)
## 설치 (Install)
    git clone https://github.com/aqntks/cv-models
    
    cd cv-models     
    
    pip install -r requirements.txt
##  모델 학습 (Model Train)
    python main.py --mode classification --model resnet --optim adam --data CIFAR_10 --batch 32 --epoch 10
--mode : 모델 유형  < classification, detection, segmentation >  
--model : 학습 모델 < mlp, cnn, resnet, alexnet, yolov5 etc >  
--optim : 옵티마이저 < adam, SGD, adagrad, RMSprop etc >  
--data : 학습 데이터 < CIFAR_10, SVHN, Places365, STL10 etc >  
--batch : 배치 사이즈  
--epoch : 세대 수  
### * 분류 모델 (Classification) *
1. MLP (Multi Layer Perceptron) - (완료)
2. CNN (Convolutional Neural Network) - (완료)
3. ResNet (Residual Neural Network) - (완료)
4. AlexNet - (완료) : 이미지 사이즈 >= 256
5. MobileNetV2 - (완료)
6. MobileNetV3 - (완료) >> version : [mobilenetv3s, mobilenetv3l]
7. DenseNet - (예정)
8. GoogleNet - (작성중)
9. Inception - (예정)
10. MnasNet - (완료) >> version : [mnasnet-0.5, mnasnet-0.75, mnasnet-1.0, mnasnet-1.3]
11. ShuffleNetV2 - (예정)
12. SqueezeNet - (완료) >> version : [squeezenet1_0, squeezenet1_1]
14. VGG - (완료) >> version : [vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn]
15. vit - (예정)
### * 탐지 모델 (Detection) *
1. yolov4 - (예정)
2. yolov5 - (예정)
3. Faster RCNN - (예정)
4. Mask R-CNN - (예정)
5. SSD - (예정)
### * 분할 모델 (Segmentation) *
1. U-Net - (예정)
2. FCN - (예정)
5. DeepLabV3 - (예정)
6. DeepLabV3+ - (예정)
7. ReSeg - (예정)
### * 생산적 적대 신경망 (GAN) *