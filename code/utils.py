import numpy as np
import torch


def np_log(x):
    return np.log(x + 1e-8)


def torch_log(x):
    return torch.log(x + 1e-8)


def calc_pred_loss(y_true, y_pred, weights):
    return - torch.mean(torch.sum(weights * y_true * torch_log(y_pred), axis=-1))


def calc_dist_loss(probs):
    avg_prob = torch.mean(probs, dim=-1)
    return torch.sum(avg_prob * torch.log(avg_prob))


def calc_clus_loss(clusters):
    pairewise_loss = - \
        torch.sum((clusters.unsqueeze(1) - clusters.unsqueeze(0)) ** 2, dim=-1)
    loss = torch.sum(pairewise_loss)

    K = clusters.shape[0]
    return loss / (K * (K-1) / 2).float()


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
