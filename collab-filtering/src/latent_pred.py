import torch
import torch.nn as nn


class LatentPred(nn.Module):
    def __init__(
            self, z_dim=1, lr=1e-4):
        super().__init__()

        self.fc = nn.Linear(z_dim, z_dim)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, z):
        z = self.fc(z)
        return z

    def step(self, z_inf, z):
        self.optim.zero_grad()
        loss = torch.sum((self(z_inf) - z)**2)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()


class LatentPredMulti(nn.Module):
    def __init__(self, v_dim, u_dim, h_dim, lr=1e-4):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(v_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, u_dim)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # define the forward computation on the data x
        hidden = self.fc1(x)
        # aggregate embeddings
        hidden = torch.sum(hidden, dim=1, keepdim=True)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        hidden = self.fc2(hidden)
        return hidden

    def step(self, z_inf, z):
        self.optim.zero_grad()
        loss = torch.sum((self(z_inf) - z)**2)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
