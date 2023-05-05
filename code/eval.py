import numpy as np
import torch
import random

from evaluation_utils import prepare_dataloader, train_loop, get_test_results, calc_metrics
from CAMELOT import CamelotModel


def eval(SEED=1001):
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = prepare_dataloader(
        SEED)

    model = CamelotModel(input_shape=(
        train_dataset.x.shape[1], train_dataset.x.shape[2]), seed=SEED, num_clusters=10, latent_dim=64)
    model = train_loop(model, train_dataset, val_dataset,
                       train_loader, val_loader, SEED=SEED)

    real, preds = get_test_results(model, test_loader)

    auc, f1, rec, nmi = calc_metrics(real, preds)

    print(f'AUCROC: \t{auc.mean():.5f}, \t{auc}')
    print(f'F1-score: \t{f1.mean():.5f}, \t{f1}')
    print(f'Recall: \t{rec.mean():.5f}, \t{rec}')
    print(f'NMI: \t\t{nmi:.5f}')


if __name__ == '__main__':
    eval()
