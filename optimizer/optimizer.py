import torch


def optim(name, model):
    if name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return optimizer