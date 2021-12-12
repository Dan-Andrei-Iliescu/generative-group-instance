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
