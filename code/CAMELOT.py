from sklearn.cluster import KMeans
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model_utils import Encoder, Identifier, Predictor, calc_l1_l2_loss
from utils import calc_pred_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def class_weight(y):
    class_numbers = torch.sum(y, dim=0)

    # Check no class is missing
    if not torch.all(class_numbers > 0):
        class_numbers += 1
    inv_class_num = 1 / class_numbers
    return inv_class_num / torch.sum(inv_class_num)


class CamelotModel(nn.Module):
    def __init__(self, input_shape, num_clusters=10, latent_dim=128, seed=12345, output_dim=4,
                 alpha=0.01, beta=0.001, regularization=(0.01, 0.01), dropout=0.0,
                 cluster_rep_lr=0.001, weighted_loss=True, attention_hidden_dim=16,
                 mlp_hidden_dim=30):

        super().__init__()
        self.seed = seed

        self.input_shape = input_shape
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.beta = beta
        self.regularization = regularization
        self.dropout = dropout
        self.cluster_rep_lr = cluster_rep_lr
        self.weighted_loss = weighted_loss
        self.attention_hidden_dim = attention_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        # three newtorks
        self.Encoder = Encoder(
            self.input_shape, self.attention_hidden_dim, self.latent_dim, self.dropout)
        self.Identifier = Identifier(
            self.latent_dim, self.mlp_hidden_dim, self.dropout, self.num_clusters)
        self.Predictor = Predictor(
            self.latent_dim, self.mlp_hidden_dim, self.dropout, self.output_dim)

        # Cluster Representation params
        self.cluster_rep_set = torch.zeros(
            size=[self.num_clusters, self.latent_dim], dtype=torch.float32, requires_grad=True)

        self.loss_weights = None

    def forward(self, x):
        z = self.Encoder(x)
        probs = self.Identifier(z)
        samples = self.get_sample(probs)
        representations = self.get_representations(samples)
        return self.Predictor(representations)

    def forward_pass(self, x):
        z = self.Encoder(x)
        probs = self.Identifier(z)
        clus_phens = self.Predictor(self.cluster_rep_set.to(device))
        y_pred = torch.matmul(probs, clus_phens)

        return y_pred, probs

    def get_sample(self, probs):
        logits = - torch.log(probs.reshape(-1, self.num_clusters))
        samples = torch.multinomial(logits, num_samples=1)
        return samples.squeeze()

    def get_representations(self, samples):
        mask = F.one_hot(samples, num_classes=self.num_clusters).to(
            torch.float32)
        return torch.matmul(mask.to(device), self.cluster_rep_set.to(device))

    def calc_pis(self, X):
        return self.Identifier(self.Encoder(X)).numpy()

    def get_cluster_reps(self):
        return self.cluster_rep_set.numpy()

    def assignment(self, X):
        pi = self.Identifier(self.Encoder(X)).numpy()
        return torch.argmax(pi, dim=1)

    def compute_cluster_phenotypes(self):
        return self.Predictor(self.cluster_rep_set).numpy()

    def initialize(self, train_data, val_data):
        x_train, y_train = train_data
        x_val, y_val = val_data
        self.loss_weights = class_weight(y_train)

        # initialize encoder
        self.initialize_encoder(x_train, y_train, x_val, y_val)

        # initialize cluster
        clus_train, clus_val = self.initialize_cluster(x_train, x_val)
        self.clus_train = clus_train
        self.x_train = x_train

        # initialize identifier
        self.initialize_identifier(x_train, clus_train, x_val, clus_val)

    def initialize_encoder(self, x_train, y_train, x_val, y_val, epochs=100, batch_size=64):
        temp = DataLoader(
            dataset=TensorDataset(x_train, y_train),
            shuffle=True,
            batch_size=batch_size
        )

        iden_loss = torch.full((epochs,), float('nan'))
        initialize_optim = torch.optim.Adam(
            self.Encoder.parameters(), lr=0.001)

        count = 0
        for i in trange(epochs):
            epoch_loss = 0
            for _, (x_batch, y_batch) in enumerate(temp):
                initialize_optim.zero_grad()

                z = self.Encoder(x_batch)
                y_pred = self.Predictor(z)
                loss = calc_pred_loss(
                    y_batch, y_pred, self.loss_weights) + calc_l1_l2_loss(part=self.Encoder)

                loss.backward(inputs=list(self.Encoder.parameters()))
                initialize_optim.step()

                epoch_loss += loss.item()

            with torch.no_grad():
                z = self.Encoder(x_val)
                y_pred_val = self.Predictor(z)
                loss_val = calc_pred_loss(y_val, y_pred_val, self.loss_weights)

            if iden_loss.min() > loss_val.item():
                torch.save(self.state_dict(), './best_model')
                count = 0
            else:
                if i < 20:
                    continue
                else:
                    count += 1
                    if count >= 20:
                        break
            iden_loss[i] = loss_val.item()

        self.load_state_dict(torch.load('./best_model'))
        print('Encoder initialization done!')

    def initialize_cluster(self, x_train, x_val):
        z = self.Encoder(x_train).cpu().detach().numpy()
        kmeans = KMeans(self.num_clusters, random_state=self.seed)
        kmeans.fit(z)
        print('Kmeans initialization done!')

        self.cluster_rep_set = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, requires_grad=True)
        train_cluster = torch.eye(self.num_clusters)[
            kmeans.predict(z)]
        val_cluster = torch.eye(self.num_clusters)[kmeans.predict(
            self.Encoder(x_val).cpu().detach().numpy())]

        print('Cluster initialization done!')
        return train_cluster, val_cluster

    def initialize_identifier(self, x_train, clus_train, x_val, clus_val, epochs=100, batch_size=64):
        temp = DataLoader(
            dataset=TensorDataset(x_train, clus_train),
            shuffle=True,
            batch_size=batch_size
        )

        iden_loss = torch.full((epochs,), float('nan'))
        initialize_optim = torch.optim.Adam(
            self.Identifier.parameters(), lr=0.001)

        count = 0
        for i in trange(epochs):
            epoch_loss = 0
            for step_, (x_batch, clus_batch) in enumerate(temp):
                initialize_optim.zero_grad()

                clus_pred = self.Identifier(self.Encoder(x_batch))
                loss = calc_pred_loss(clus_batch, clus_pred) + \
                    calc_l1_l2_loss(layers=[self.Identifier.fc2])

                loss.backward(inputs=list(self.Identifier.parameters()))
                initialize_optim.step()

                epoch_loss += loss.item()

            with torch.no_grad():
                clus_pred_val = self.Identifier(self.Encoder(x_val))
                loss_val = calc_pred_loss(clus_val, clus_pred_val)

            if iden_loss.min() > loss_val.item():
                torch.save(self.state_dict(), './best_model')
                count = 0
            else:
                if i < 20:
                    continue
                else:
                    count += 1
                    if count >= 20:
                        break
            iden_loss[i] = loss_val.item()

        self.load_state_dict(torch.load('./best_model'))
        print('Identifier initialization done!')
