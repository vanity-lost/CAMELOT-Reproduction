import numpy as np
import torch
import random
import os

from evaluation_utils import prepare_dataloader, train_loop
from CAMELOT import CamelotModel

if os.path.exists('best_model'):
    os.remove('best_model')


def train(SEED=1012):
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = prepare_dataloader(
        SEED)

    model = CamelotModel(input_shape=(train_dataset.x.shape[1],
                                      train_dataset.x.shape[2]),
                         seed=SEED,
                         num_clusters=10,
                         latent_dim=64)
    model = train_loop(model,
                       train_dataset,
                       val_dataset,
                       train_loader,
                       val_loader,
                       SEED=SEED)

    torch.save(model.state_dict(), 'best_model')


if __name__ == '__main__':
    train()
