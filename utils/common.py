import torch


def model_save(model, state_dict=False):
    if state_dict:
        torch.save(model.state_dict(), 'model.pth')
    else:
        torch.save(model, 'model.pth')
