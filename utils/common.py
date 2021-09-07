import torch


def model_save(model, state_dict=False):
    if state_dict:
        torch.save(model.state_dict(), 'result/model.pt')
    else:
        torch.save(model, 'result/model.pt')


def lightWeight(model):
    model['epoch'] = -1
    model['model'].half()  # to FP16
    for p in model['model'].parameters():
        p.requires_grad = False