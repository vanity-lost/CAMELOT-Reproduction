import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatTimeAttention(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super().__init__()

        self.latent_dim = latent_dim
        N, T, D_f = input_shape
        # Define Kernel and Bias for Feature Projection
        self.kernel = torch.zeros(
            (1, 1, D_f, self.latent_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.kernel)
        self.bias = torch.zeros(
            (1, 1, D_f, self.latent_dim), requires_grad=True)
        nn.init.uniform_(self.bias)

        # Define Time aggregation weights for averaging over time.
        self.unnorm_beta = torch.zeros((1, T, 1), requires_grad=True)
        nn.init.uniform_(self.unnorm_beta)

    def forward(self, x, latent):
        o_hat, _ = self.generate_latent_approx(x, latent)
        weights = self.calc_weights(self.unnorm_beta)
        return torch.sum(torch.bmm(o_hat, weights), dim=1)

    def generate_latent_approx(self, x, latent):
        features = torch.mul(x.unsqueeze(-1), self.kernel) + self.bias
        features = F.relu(features)

        # calculate the score
        score_hat = torch.matmul(torch.inverse(torch.matmul(features, features.transpose(
            1, -1))), torch.matmul(features, latent.unsqueeze(-1)))
        scores = torch.squeeze(score_hat)

        o_hat = torch.sum(torch.bmm(scores.unsqueeze(-1), features), dim=2)

        return o_hat, scores

    def calc_weights(self, x):
        abs_x = torch.abs(x)
        return abs_x / torch.sum(abs_x, dim=1)
