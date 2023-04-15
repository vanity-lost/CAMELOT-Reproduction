from sklearn.cluster import KMeans
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model_utils import Encoder, Identifier, Predictor
from utils import calc_pred_loss


class CamelotModel(nn.Module):
    def __init__(self, input_shape, num_clusters=10, latent_dim=32, seed=4347, output_dim=4,
                 alpha=0.01, beta=0.01, regularization=(0.01, 0.01), dropout=0.6,
                 cluster_rep_lr=0.001, weighted_loss=True, attention_hidden_dim=20,
                 mlp_hidden_dim=30):

        super().__init__()
        torch.random.manual_seed(seed)
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
        self.Encoder = Encoder(self.input_shape, self.attention_hidden_dim, self.latent_dim, self.dropout)
        self.Identifier = Identifier(self.latent_dim, self.mlp_hidden_dim, self.dropout, self.num_clusters)
        self.Predictor = Predictor(self.num_clusters, self.mlp_hidden_dim, self.dropout, self.output_dim)

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
        samples = self.get_sample(probs)
        representations = self.get_representations(samples)
        return self.Predictor(representations), probs

    def get_sample(self, probs):
        logits = torch.log(probs.reshape(-1, self.num_clusters))
        samples = torch.multinomial(logits, num_samples=1)
        return samples.squeeze()

    def get_representations(self, samples):
        mask = F.one_hot(samples, num_classes=self.num_clusters)
        return mask @ self.cluster_rep_set

    def class_weight(self, y):
        inv_class_num = 1 / torch.sum(y, dim=0)
        return inv_class_num / torch.sum(inv_class_num)

    def calc_pis(self, X):
        return self.Identifier(self.Encoder(X)).numpy()

    def get_cluster_reps(self):
        return self.cluster_rep_set.numpy()

    def assignment(self, X):
        pi = self.Identifier(self.Encoder(X)).numpy()
        return torch.argmax(pi, dim=1)

    def compute_cluster_phenotypes(self):
        return self.Predictor(self.cluster_rep_set).numpy()

    # def compute_unnorm_attention_weights(self, inputs):
    #     # no idea
    #     return self.Encoder.compute_unnorm_scores(inputs, cluster_reps=self.cluster_rep_set)

    # def compute_norm_attention_weights(self, inputs):
    #     # no idea
    #     return self.Encoder.compute_norm_scores(inputs, cluster_reps=self.cluster_rep_set)

    def initialize(self, train_data, val_data):
        x_train, y_train = train_data
        x_val, y_val = val_data
        self.loss_weights = self.class_weight(y_train)

        # initialize encoder
        self.initialize_encoder(x_train, y_train, x_val, y_val)

        # initialize cluster
        clus_train, clus_val = self.initialize_cluster(x_train, x_val)
        self.clus_train = clus_train
        self.x_train = x_train

        # initialize identifier
        self.initialize_identifier(x_train, clus_train, x_val, clus_val)

    def initialize_encoder(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=64):
        temp = DataLoader(
            dataset=TensorDataset(x_train, y_train),
            shuffle=True,
            batch_size=batch_size
        )

        iden_loss = torch.full((epochs,), float('nan'))
        initialize_optim = torch.optim.Adam(
            self.Encoder.parameters(), lr=0.001)

        for i in trange(epochs):
            epoch_loss = 0
            for _, (x_batch, y_batch) in enumerate(temp):
                initialize_optim.zero_grad()
                
                z = self.Encoder(x_batch)
                y_pred = self.Predictor(self.Identifier(z))
                loss = calc_pred_loss(y_batch, y_pred, self.loss_weights)

                loss.backward()
                initialize_optim.step()

                epoch_loss += loss.item()

            with torch.no_grad():
                z = self.Encoder(x_val)
                y_pred_val = self.Predictor(self.Identifier(z))
                loss_val = calc_pred_loss(y_val, y_pred_val, self.loss_weights)

            iden_loss[i] = loss_val.item()
            if torch.le(iden_loss[-50:], loss_val.item() + 0.001).any():
                break

        print('Encoder initialization done!')

    def initialize_cluster(self, x_train, x_val):
        z = self.Encoder(x_train).detach().numpy()
        kmeans = KMeans(self.num_clusters, random_state=self.seed)
        kmeans.fit(z)
        print('Kmeans initialization done!')

        self.cluster_rep_set = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32)
        train_cluster = torch.eye(self.num_clusters)[
            kmeans.predict(z)]
        val_cluster = torch.eye(self.num_clusters)[kmeans.predict(
            self.Encoder(x_val).detach().numpy())]
        
        print('Cluster initialization done!')
        return train_cluster, val_cluster

    def initialize_identifier(self, x_train, clus_train, x_val, clus_val, epochs=10, batch_size=64):
        temp = DataLoader(
            dataset=TensorDataset(x_train, clus_train),
            shuffle=True,
            batch_size=batch_size
        )

        iden_loss = torch.full((epochs,), float('nan'))
        initialize_optim = torch.optim.Adam(
            self.Identifier.parameters(), lr=0.001)

        for i in trange(epochs):
            epoch_loss = 0
            for step_, (x_batch, clus_batch) in enumerate(temp):
                initialize_optim.zero_grad()

                clus_pred = self.Identifier(self.Encoder(x_batch))
                print(clus_pred.shape, clus_batch.shape)
                loss = calc_pred_loss(clus_batch, clus_pred, self.loss_weights)

                loss.backward()
                initialize_optim.step()

                epoch_loss += loss.item()

            with torch.no_grad():
                clus_pred_val = self.Identifier(self.Encoder(x_val))
                print(clus_val.shape, clus_pred_val.shape)
                loss_val = calc_pred_loss(clus_val, clus_pred_val, self.loss_weights)

            iden_loss[i] = loss_val.item()
            if torch.le(iden_loss[-50:], loss_val.item() + 0.001).any():
                break

        print('Identifier initialization done!')
