import torch
import argparse

from data.load_data import data
from utils.common import model_save

from selector.select import classification


def main(opt):
    mode, model_name, optimizer, img_data, batch_size, epochs \
        = opt.mode, opt.model, opt.optim, opt.data, opt.batch, opt.epoch

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('PyTorch 버전:', torch.__version__, ' Device:', device)

    train_dataset, test_dataset = data(img_data)

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

    model = classification(model_name, optimizer, device, epochs, train_loader, test_loader)

    model_save(model, state_dict=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='classification')
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--data', type=str, default='CIFAR_10')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    opt = parser.parse_args()
    main(opt)
