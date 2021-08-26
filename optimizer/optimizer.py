import torch


def optim(name, model, lr, momentum):
    if name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if name == "AdaGrad":
        optimizer = torch.optim.Adagrad(model.parameters())
    if name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters())

    return optimizer