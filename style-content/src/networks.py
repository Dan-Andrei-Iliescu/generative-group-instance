import torch
import torch.nn as nn


class InstEncoder(nn.Module):
    def __init__(self, x_dim, r_dim, u_dim, v_dim, h_dim):
        super().__init__()
        self.w1 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(r_dim, u_dim, h_dim)))
        self.w2 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(x_dim, v_dim, h_dim)))
        self.relu = torch.nn.ReLU()

    def forward(self, x, r, u):
        v = torch.relu(torch.einsum(
            'ruh,nijr,nruv->nijh', self.w1, r, u))
        v = torch.einsum('xvh,nijx,nijh->nijv', self.w2, x, v)
        return v


class GroupEncoder(nn.Module):
    def __init__(self, x_dim, r_dim, u_dim, v_dim, h_dim):
        super().__init__()
        self.w1 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(x_dim, r_dim, v_dim, h_dim)))
        self.w2 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(r_dim, u_dim, v_dim, h_dim)))
        self.relu = torch.nn.ReLU()

    def forward(self, x, r):
        norm = x.shape[1] * x.shape[2]
        u = torch.relu(torch.einsum(
            'xrvh,nijx,nijr->nrvh', self.w1, x, r) / norm)
        u = torch.einsum('ruvh,nrvh->nruv', self.w2, u)
        return u


class Decoder(nn.Module):
    def __init__(self, x_dim, r_dim, u_dim, v_dim, h_dim):
        super().__init__()
        self.w1 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(r_dim, u_dim, v_dim, h_dim)))
        self.w2 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(x_dim, r_dim, h_dim)))
        self.relu = torch.nn.ReLU()

    def forward(self, r, u, v):
        x = self.relu(torch.einsum('ruvh,nruv,nijv->nijrh', self.w1, u, v))
        x = torch.einsum('xrh,nijr,nijrh->nijx', self.w2, r, x)
        return x
