import torch
import torch.nn as nn


class InstEncoder(nn.Module):
    def __init__(self, x_dim, r_dim, u_dim, v_dim, h_dim):
        super().__init__()
        self.w1 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(r_dim, u_dim, v_dim, h_dim)))
        self.w2 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(x_dim, v_dim, h_dim)))
        self.relu = torch.nn.ReLU()

    def forward(self, x, r, u):
        v = torch.einsum('ruvh,nruv->nrh', self.w1, u)
        v = torch.einsum('nijr,nrh->nijh', r, v)
        v = torch.relu(v)
        v = torch.einsum('xvh,nijh->nijxv', self.w2, v)
        v = torch.einsum('nijx,nijxv->nijv', x, v)
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
        u = torch.einsum('nijx,nijr->nxr', x, r) / norm
        u = torch.einsum('xrvh,nxr->nrvh', self.w1, u)
        u = torch.relu(u)
        u = torch.einsum('ruvh,nrvh->nruv', self.w2, u)
        return u


class Decoder(nn.Module):
    def __init__(self, x_dim, r_dim, u_dim, v_dim, h_dim):
        super().__init__()
        self.w1 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(u_dim, h_dim)))
        self.w2 = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(
            torch.randn(x_dim, r_dim, h_dim)))
        self.relu = torch.nn.ReLU()

    def forward(self, r, u, v):
        x = torch.einsum('nruv,nijv->nijru', u, v)
        x = torch.einsum('nijru,nijr->nijru', x, r)
        x = torch.einsum('uh,nijru->nijrh', self.w1, x)
        x = torch.relu(x)
        x = torch.einsum('xrh,nijrh->nijx', self.w2, x)
        return x
