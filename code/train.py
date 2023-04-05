import torch.nn as nn

def mix_l1_l2_reg(l1_ratio, l2_ratio):
    def regularizer(model):
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        return l1_ratio * l1_norm + l2_ratio * l2_norm
    return regularizer

l1_ratio = 0.01
l2_ratio = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=mix_l1_l2_reg(l1_ratio, l2_ratio))
