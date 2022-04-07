import torch
import torch.nn as nn

from src.networks import InstEncoder, GroupEncoder
from src.networks import Decoder
from utils.helpers import bin_pos_emb, trig_pos_emb


class Model(nn.Module):
    def __init__(
            self, x_dim=3, r_dim=16, u_dim=8, v_dim=6, h_dim=16, lr=1e-2):
        super().__init__()
        # The image is batch_size x height x width x scales x features
        # Parameters
        self.x_dim = x_dim
        self.r_dim = r_dim
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.h_dim = h_dim
        args = (self.x_dim, self.r_dim, self.u_dim, self.v_dim, self.h_dim)

        # Networks
        self.group_enc = GroupEncoder(*args)
        self.inst_enc = InstEncoder(*args)
        self.decoder = Decoder(*args)

        # Optimizer
        self.model_params = list(self.group_enc.parameters()) + \
            list(self.inst_enc.parameters()) + \
            list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(
            self.model_params, lr=lr)

    # define the variational posterior q(u|{x}) \prod_i q(v_i|x_i,u)
    def inference(self, x):
        # Positional embedding
        r = bin_pos_emb(x, self.r_dim)

        # Sample q(u|{x})
        u = self.group_enc.forward(x, r)

        # Sample q(v|x,u)
        v = self.inst_enc.forward(x, r, u)

        # Turn vector of weights into one hot using hard max
        _, idcs = v.max(dim=-1)
        idcs = idcs.unsqueeze(-1).broadcast_to(v.shape)
        v_hard = torch.zeros_like(v)
        v_hard.scatter_(-1, idcs, 1)

        # Straight-Through trick
        v_hard = (v_hard - v).detach() + v

        return u, v_hard, v

    def generate(self, u, v):
        # Concatenate positional embedding
        r = bin_pos_emb(v, self.r_dim)

        # Generative data dist
        x_loc = self.decoder.forward(r, u, v)

        return x_loc

    # ELBO loss for hierarchical variational autoencoder
    def elbo_func(self, x):
        # Latent inference
        u, v_hard, v = self.inference(x)

        # Generative data dist
        x_loc = self.generate(u, v_hard)

        # Losses
        kl_u = torch.mean(u**2)
        kl_v = torch.mean(v**2)
        lik = torch.mean(torch.abs(x - x_loc))
        return lik + kl_u + kl_v

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        # sample q(u,{v}|{x})
        u, v, _ = self.inference(x)
        # decode p({x}|u,{v})
        x_loc = self.generate(u, v)
        return x_loc, v

    # define a helper function for translating images
    def translate(self, x, y):
        # sample q(u,{v}|{x})
        _, v, _ = self.inference(x)
        # sample q(uy|{y})
        u, _, _ = self.inference(y)
        # decode p({x}|uy,{v})
        trans = self.generate(u, v)
        return trans

    # one training step
    def step(self, x):
        self.optimizer.zero_grad()
        loss = self.elbo_func(x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
