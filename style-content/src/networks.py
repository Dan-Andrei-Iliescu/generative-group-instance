import torch
import torch.nn as nn


class MoveConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = x.movedim([1, 2, 3], [2, 3, 1])
        x = self.conv(x)
        x = x.movedim([1, 2, 3], [3, 1, 2])
        return x


class InstEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        h_dim = 32
        self.conv1 = MoveConv(in_dim, out_dim)
        self.conv2 = MoveConv(h_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, u):
        u = u.expand(-1, x.shape[1], x.shape[2], -1)
        x = torch.cat([x, u], dim=-1)
        x = self.conv1(x)
        return x


class GroupEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        h_dim = 64
        self.conv1 = MoveConv(in_dim, h_dim)
        self.conv2 = MoveConv(h_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.mean(x, dim=[1, 2], keepdim=True)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        h_dim = 32
        self.conv1 = MoveConv(in_dim, h_dim)
        self.conv2 = MoveConv(h_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, u, v):
        u = u.expand(-1, v.shape[1], v.shape[2], -1)
        x = torch.cat([u, v], dim=-1)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x
