import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical, RelaxedOneHotCategorical

from src.networks import InstEncoder, GroupEncoder
from src.networks import Decoder
from utils.helpers import bin_pos_emb, trig_pos_emb


class Model(nn.Module):
    def __init__(
            self, x_dim=3, u_dim=4, v_dim=3, num_bits=32, lr=1e-2):
        super().__init__()
        # The image is batch_size x height x width x scales x features
        # Parameters
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.num_bits = num_bits

        # Networks
        self.group_enc = GroupEncoder(
            in_dim=x_dim+4*num_bits, out_dim=v_dim*u_dim)
        self.inst_enc = InstEncoder(
            in_dim=x_dim+u_dim+4*num_bits, out_dim=v_dim)
        self.decoder = Decoder(in_dim=u_dim+v_dim+4*num_bits, out_dim=x_dim)

        # setup the optimizer
        self.model_params = list(self.group_enc.parameters()) + \
            list(self.inst_enc.parameters()) + \
            list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(self.model_params, lr=lr)

    def prior(self, u, v):
        pu = Normal(torch.zeros_like(u), torch.ones_like(u))
        pv = OneHotCategorical(logits=torch.ones_like(v))
        return pu, pv

    # define the variational posterior q(u|{x}) \prod_i q(v_i|x_i,u)
    def inference(self, x):
        v = torch.zeros_like(x)

        # Concatenate positional embedding
        x = trig_pos_emb(x, self.num_bits)

        # Sample q(u|{x})
        u = self.group_enc.forward(x)

        # Sample q(v|x,u)
        v = self.inst_enc.forward(x, u)

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
        v = trig_pos_emb(v, self.num_bits)

        # Generative data dist
        x_loc = self.decoder.forward(u, v)

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
        return lik + kl_v + kl_u

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


if __name__ == '__main__':
    model = Model()

    x = torch.rand(size=[16, 128, 64, 5])
    pos = model.pos_encoding(x, 7)
    print(pos)
