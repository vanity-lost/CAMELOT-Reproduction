import numpy as np
import random
import torch
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score

from data_utils import CustomDataset, load_data
from train_utils import MyLRScheduler, class_weight, calc_pred_loss, calc_clus_loss, calc_dist_loss, calc_l1_l2_loss


def prepare_dataloader(SEED=12345):
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    dataset = CustomDataset(time_range=(0, 10))

    # Stratified Sampling for train and val
    train_idx, test_idx = train_test_split(np.arange(len(dataset)),
                                           test_size=0.4,
                                           random_state=SEED,
                                           shuffle=True,
                                           stratify=np.argmax(dataset.y, axis=-1))

    # Subset dataset for train and val
    train_val_dataset = dataset.get_subset(train_idx)
    test_dataset = dataset.get_subset(test_idx)

    train_idx, val_idx = train_test_split(np.arange(len(train_val_dataset)),
                                          test_size=0.4,
                                          random_state=SEED,
                                          shuffle=True,
                                          stratify=np.argmax(train_val_dataset.y, axis=-1))

    train_dataset = train_val_dataset.get_subset(train_idx)
    val_dataset = train_val_dataset.get_subset(val_idx)

    train_loader, val_loader, test_loader = load_data(
        train_dataset, val_dataset, test_dataset)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def train_loop(model, train_dataset, val_dataset, train_loader, val_loader, SEED=12345):
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    train_x = torch.tensor(train_dataset.x).to(device)
    train_y = torch.tensor(train_dataset.y).to(device)
    val_x = torch.tensor(val_dataset.x).to(device)
    val_y = torch.tensor(val_dataset.y).to(device)

    model.initialize((train_x, train_y), (val_x, val_y))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cluster_optim = torch.optim.Adam([model.cluster_rep_set], lr=0.001)

    lr_scheduler = MyLRScheduler(
        optimizer, patience=15, min_lr=0.00001, factor=0.25)
    cluster_lr_scheduler = MyLRScheduler(
        cluster_optim, patience=15, min_lr=0.00001, factor=0.25)

    loss_mat = np.zeros((100, 4, 2))

    best_loss = 1e5
    count = 0
    for i in trange(100):
        for step, (x_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad()
            cluster_optim.zero_grad()

            y_pred, probs = model.forward_pass(x_train)

            loss_weights = class_weight(y_train)

            common_loss = calc_pred_loss(y_train, y_pred, loss_weights)

            enc_loss = common_loss + model.alpha * calc_dist_loss(probs) + \
                + calc_l1_l2_loss(part=model.Encoder)
            enc_loss.backward(retain_graph=True, inputs=list(
                model.Encoder.parameters()))

            idnetifier_loss = common_loss + model.alpha * calc_dist_loss(probs) + \
                + calc_l1_l2_loss(layers=[model.Identifier.fc2])
            idnetifier_loss.backward(
                retain_graph=True, inputs=list(model.Identifier.parameters()))

            pred_loss = common_loss + \
                calc_l1_l2_loss(
                    layers=[model.Predictor.fc2, model.Predictor.fc3])
            pred_loss.backward(retain_graph=True, inputs=list(
                model.Predictor.parameters()))

            clus_loss = common_loss + model.beta * \
                calc_clus_loss(model.cluster_rep_set)
            clus_loss.backward(inputs=model.cluster_rep_set)

            optimizer.step()
            cluster_optim.step()

            loss_mat[i, 0, 0] += enc_loss.item()
            loss_mat[i, 1, 0] += idnetifier_loss.item()
            loss_mat[i, 2, 0] += pred_loss.item()
            loss_mat[i, 3, 0] += clus_loss.item()

        with torch.no_grad():
            for step, (x_val, y_val) in enumerate(val_loader):
                y_pred, probs = model.forward_pass(x_val)

                loss_weights = class_weight(y_val)

                common_loss = calc_pred_loss(y_val, y_pred, loss_weights)

                enc_loss = common_loss + model.alpha * calc_dist_loss(probs) + \
                    + calc_l1_l2_loss(part=model.Encoder)

                idnetifier_loss = common_loss + model.alpha * calc_dist_loss(probs) + \
                    + calc_l1_l2_loss(layers=[model.Identifier.fc2])

                pred_loss = common_loss + \
                    calc_l1_l2_loss(
                        layers=[model.Predictor.fc2, model.Predictor.fc3])

                clus_loss = common_loss + model.beta * \
                    calc_clus_loss(model.cluster_rep_set)

                loss_mat[i, 0, 1] += enc_loss.item()
                loss_mat[i, 1, 1] += idnetifier_loss.item()
                loss_mat[i, 2, 1] += pred_loss.item()
                loss_mat[i, 3, 1] += clus_loss.item()

            if i >= 30:
                if loss_mat[i, 0, 1] < best_loss:
                    count = 0
                    best_loss = loss_mat[i, 0, 1]
                    torch.save(model.state_dict(), 'best_model')
                else:
                    count += 1
                    if count >= 50:
                        model.load_state_dict(torch.load('best_model'))
        lr_scheduler.step(loss_mat[i, 0, 1])
        cluster_lr_scheduler.step(loss_mat[i, 0, 1])

    model.load_state_dict(torch.load('best_model'))
    return model


def get_test_results(model, test_loader):
    real, preds = [], []
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            y_pred, _ = model.forward_pass(x)
            preds.extend(list(y_pred.cpu().detach().numpy()))
            real.extend(list(y.cpu().detach().numpy()))
    return real, preds


def calc_metrics(real, preds):
    auc = roc_auc_score(real, preds, average=None)

    labels_true, labels_pred = np.argmax(
        real, axis=1), np.argmax(preds, axis=1)

    # Compute F1
    f1 = f1_score(labels_true, labels_pred, average=None)

    # Compute Recall
    rec = recall_score(labels_true, labels_pred, average=None)

    # Compute NMI
    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    return auc, f1, rec, nmi
