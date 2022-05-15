import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, z_dim, h_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(1, h_dim)
        self.fc2m = nn.Linear(h_dim, z_dim)
        self.fc2s = nn.Linear(h_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        # define the forward computation on the data x
        h = self.relu(self.fc1(x - 4.))
        h = self.drop(h)
        z_loc = self.fc2m(h)
        z_scale = self.softplus(self.fc2s(h))
        return z_loc, z_scale


class InstEncoder(nn.Module):
    def __init__(self, z_dim, h_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(z_dim+1, h_dim)
        self.fc2m = nn.Linear(h_dim, z_dim)
        self.fc2s = nn.Linear(h_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x, u):
        u = torch.broadcast_to(u, [x.shape[0], u.shape[1]])
        x = torch.cat([x - 4., u], dim=-1)
        h = self.relu(self.fc1(x))
        h = self.drop(h)
        v_loc = self.fc2m(h)
        v_scale = self.softplus(self.fc2s(h))
        return v_loc, v_scale


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            name='bias', param=torch.nn.Parameter(torch.randn(1)))
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, u, v):
        x = 4 + u[:, 0] + v[:, 0] + \
            torch.sum(u[:, 1:] * v[:, 1:], dim=-1)
        return x.reshape(-1, 1)
