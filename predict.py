import cv2
import numpy as np
import torch
import argparse
from selector.select import get_classification
from selector.cls_train import test
from torchvision import transforms
from PIL import Image


def predict(opt):
    model_name, weight, img, labels = opt.model, opt.pt, opt.img, opt.label

    label = []
    f = open('data/label.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        label.append(line)
    f.close()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('PyTorch 버전:', torch.__version__, ' Device:', device)

    model = get_classification(model_name, device, 0, 0, 0, 0)
    model.load_state_dict(torch.load(weight, map_location=device))

    print(model)

    transform = transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = Image.open(img)
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)

    pred = test(model, batch_t, device)

    print('예측 결과:', label[int(pred)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--pt', type=str, default='result/resnet.pt')
    parser.add_argument('--img', type=str, default='data/test.jpg')
    parser.add_argument('--label', type=str, default='data/label.txt')
    opt = parser.parse_args()
    predict(opt)