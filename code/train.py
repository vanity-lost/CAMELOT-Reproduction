from tqdm import tqdm, trange
import torch

from CAMELOT import CamelotModel
from datasets_dataloader_pytorch import CustomDataset, load_data
from utils import calc_pred_loss, calc_dist_loss, calc_clus_loss

epochs = 100

dataset = CustomDataset()
train_data = load_data(dataset)
val_data = load_data(dataset)

input_shape = (1, )

model = CamelotModel(input_shape)
model.initialize(train_data, val_data)

encoder_optim = torch.optim.Adam(model.Identifier.parameters(), lr=0.001)
identifier_optim = torch.optim.Adam(model.Identifier.parameters(), lr=0.001)
predictor_optim = torch.optim.Adam(model.Identifier.parameters(), lr=0.001)
cluster_optim = torch.optim.Adam(model.Identifier.parameters(), lr=0.001)

iden_loss = []

for i in range(epochs):
    epoch_pred_loss, epoch_enc_id_loss, epoch_clus_loss = 0, 0, 0
    for step_, (x_train, y_train) in enumerate(train_data):
        encoder_optim.zero_grad()
        identifier_optim.zero_grad()
        predictor_optim.zero_grad()
        cluster_optim.zero_grad()

        y_pred, probs = model.forward_pass(x_train)
        pred_loss = calc_pred_loss(y_train, y_pred, model.loss_weights)
        enc_id_loss = calc_pred_loss(
            y_train, y_pred, model.loss_weights) + model.alpha * calc_dist_loss(probs)
        clus_loss = calc_pred_loss(y_train, y_pred, model.loss_weights) + \
            model.beta * calc_clus_loss(model.cluster_rep_set)

        pred_loss.backward()
        enc_id_loss.backward()
        clus_loss.backward()

        encoder_optim.step()
        identifier_optim.step()
        predictor_optim.step()
        cluster_optim.step()

        epoch_pred_loss += pred_loss.item()
        epoch_enc_id_loss += enc_id_loss.item()
        epoch_clus_loss += clus_loss.item()
