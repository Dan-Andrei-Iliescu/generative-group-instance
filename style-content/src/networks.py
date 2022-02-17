import torch
import torch.nn as nn


class MoveConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = x.movedim([1, 2, 3, 4], [3, 4, 2, 1])
        x = self.conv(x)
        x = x.movedim([1, 2, 3, 4], [4, 3, 1, 2])
        return x


class InstEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = MoveConv(in_dim, out_dim)

    def forward(self, x, u):
        u = u.expand(-1, x.shape[1], x.shape[2], -1, -1)
        u = torch.zeros_like(u)
        x = torch.cat([x, u], dim=-1)
        x = self.conv(x)
        return x


class GroupEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = MoveConv(in_dim, out_dim)
        self.conv2 = MoveConv(in_dim, out_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        mu = self.conv1(x)
        sig = self.conv2(x)
        mu = torch.mean(mu, dim=[1, 2], keepdim=True)
        sig = self.softplus(torch.mean(sig, dim=[1, 2], keepdim=True))
        return mu, sig


class DecoderParallel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = MoveConv(in_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, u, v):
        u = u.expand(-1, v.shape[1], v.shape[2], -1, -1)
        x = torch.cat([u, v], dim=-1)
        x = self.conv(x)
        x = torch.mean(x, dim=[3])
        # x = self.tanh(x)
        return x
