import numpy as np
import torch


def np_log(x):
    return np.log(x + 1e-8)


def torch_log(x):
    return torch.log(x + 1e-8)


def clus_pred_loss(y_true, y_pred, weights):
    return - torch.mean(weights * y_true * torch_log(y_pred), dim=-1)


# import torch.nn as nn

# def mix_l1_l2_reg(l1_ratio, l2_ratio):
#     def regularizer(model):
#         l1_norm = sum(p.abs().sum() for p in model.parameters())
#         l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
#         return l1_ratio * l1_norm + l2_ratio * l2_norm
#     return regularizer

# l1_ratio = 0.01
# l2_ratio = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=mix_l1_l2_reg(l1_ratio, l2_ratio))
