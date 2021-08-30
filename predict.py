import cv2
import numpy as np
import torch
import argparse
from selector.select import get_classification
from selector.cls_train import test
from torchvision import transforms
from PIL import Image




def predict(opt):
    model_name, weight, img = opt.model, opt.pt, opt.img

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('PyTorch 버전:', torch.__version__, ' Device:', device)

    model = get_classification(model_name, device, 0, 0, 0, 0)
    model.load_state_dict(torch.load(weight, map_location=device))

    # transform = transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = Image.open(img)
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)

    test(model, batch_t, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--pt', type=str, default='result/example.pt')
    parser.add_argument('--img', type=str, default='data/test.jpg')
    opt = parser.parse_args()
    predict(opt)