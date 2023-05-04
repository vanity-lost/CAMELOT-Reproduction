import torch


class MyLRScheduler():
    def __init__(self, optimizer, patience, min_lr, factor):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.wait = 0
        self.best_loss = float('inf')

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr


def calc_l1_l2_loss(part=None, layers=None):
    parameters = []
    if part:
        for parameter in part.parameters():
            parameters.append(parameter.view(-1))
        parameters = torch.cat(parameters)
    elif layers:
        for layer in layers:
            parameters.extend(layer.parameters())
        parameters = torch.cat([p.view(-1) for p in parameters])
    else:
        parameters = torch.tensor(parameters)
    return 1e-30 * torch.abs(parameters).sum() + 1e-30 * torch.square(parameters).sum()


def class_weight(y):
    class_numbers = torch.sum(y, dim=0)

    # Check no class is missing
    if not torch.all(class_numbers > 0):
        class_numbers += 1
    inv_class_num = 1 / class_numbers
    return inv_class_num / torch.sum(inv_class_num)


def torch_log(x):
    return torch.log(x + 1e-8)


def calc_pred_loss(y_true, y_pred, weights=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if weights is None:
        weights = torch.ones(y_true.shape) / y_true.shape[-1]
    return - torch.mean(torch.sum(weights.to(device) * y_true.to(device) * torch_log(y_pred).to(device), dim=-1))


def calc_dist_loss(probs):
    avg_prob = torch.mean(probs, dim=-1)
    return torch.sum(avg_prob * torch.log(avg_prob))


def calc_clus_loss(clusters):
    pairewise_loss = - \
        torch.sum((clusters.unsqueeze(1) - clusters.unsqueeze(0)) ** 2, dim=-1)
    loss = torch.sum(pairewise_loss)

    K = clusters.shape[0]
    return (loss / (K * (K - 1) / 2)).float()
