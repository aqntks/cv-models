import torch


def optim(name, model):
    if name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if name == "AdaGrad":
        optimizer = torch.optim.Adagrad(model.parameters())
    if name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters())

    return optimizer