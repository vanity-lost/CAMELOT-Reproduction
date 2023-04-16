import numpy as np
import torch


def np_log(x):
    return np.log(x + 1e-8)


def torch_log(x):
    return torch.log(x + 1e-8)


def calc_pred_loss(y_true, y_pred, weights=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if weights is None:
        weights = torch.ones(y_true.shape) / y_true.shape[-1]
    return - torch.mean(torch.sum(weights.to(device) * y_true.to(device) * torch_log(y_pred).to(device), axis=-1))


def calc_dist_loss(probs):
    avg_prob = torch.mean(probs, dim=-1)
    return torch.sum(avg_prob * torch.log(avg_prob))


def calc_clus_loss(clusters):
    pairewise_loss = - \
        torch.sum((clusters.unsqueeze(1) - clusters.unsqueeze(0)) ** 2, dim=-1)
    loss = torch.sum(pairewise_loss)

    K = clusters.shape[0]
    return (loss / (K * (K-1) / 2)).float()
