import torch
import argparse
from selector.select import get_classification
from selector.cls_train import test


def predict(opt):
    model_name, weight, img = opt.model, opt.pt, opt.img

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('PyTorch 버전:', torch.__version__, ' Device:', device)

    model = get_classification(model_name, device)
    model = model.load_state_dict(weight)

    test(model, image, label, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--pt', type=str, default='result/example.pt')
    parser.add_argument('--img', type=str, default='adam')
    opt = parser.parse_args()
    predict(opt)