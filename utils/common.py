import torch


def model_save(model, state_dict=False):
    if state_dict:
        torch.save(model.state_dict(), 'result/model.pt')
    else:
        torch.save(model, 'result/model.pt')
