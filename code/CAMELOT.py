from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import FeatTimeAttention


class CamelotModel(nn.Module):
    def __init__(self, input_shape, num_clusters=10, latent_dim=32, seed=4347, output_dim=4,
                 alpha=0.01, beta=0.01, regularization=(0.01, 0.01), dropout=0.6,
                 cluster_rep_lr=0.001, weighted_loss=True, attention_hidden_dim=20,
                 mlp_hidden_dim=30):

        super().__init__()
        torch.random.manual_seed(seed)

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
        self.Encoder = nn.Sequential(
            # TODO: check dropout, input_dim
            nn.LSTM(1, self.attention_hidden_dim,
                    num_layers=3, dropout=self.dropout),
            FeatTimeAttention(self.latent_dim, self.input_shape),
        )
        self.Identifier = nn.Sequential(
            # TODO: check regularization, input_dim
            nn.Linear(1, self.mlp_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_hidden_dim, self.output_dim),
            nn.Softmax(),
        )
        self.Predictor = nn.Sequential(
            # TODO: check regularization, input_dim
            nn.Linear(1, self.mlp_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_hidden_dim, self.output_dim),
            nn.Softmax(),
        )

        # Cluster Representation params
        self.cluster_rep_set = torch.zeros(
            size=[self.num_clusters, self.latent_dim], dtype=torch.float32, requires_grad=True)

        self.loss_weights = None

    def forward(self, x):
        z = self.Encoder(x)
        probs = self.Identifier(z)
        samples = self.generate_sample(probs)
        representations = self.generate_representations(samples)
        return self.Predictor(representations)

    def generate_sample(self, probs):
        logits = torch.log(probs.reshape(-1, self.num_clusters))
        samples = torch.multinomial(logits, num_samples=1)
        return samples.squeeze()

    def generate_representations(self, samples):
        mask = F.one_hot(samples, num_classes=self.num_clusters)
        return mask @ self.cluster_rep_set

    # TODO: initialize the model
