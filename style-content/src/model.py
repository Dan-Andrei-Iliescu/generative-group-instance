import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical, RelaxedOneHotCategorical

from src.networks import InstEncoder, GroupEncoder
from src.networks import DecoderParallel as Decoder
from utils.helpers import bin_pos_emb


class Model(nn.Module):
    def __init__(
            self, x_dim=3, u_dim=16, v_dim=5, r_dim=2, num_bits=1, lr=1e-0):
        super().__init__()
        # The image is batch_size x height x width x scales x features
        # Parameters
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.r_dim = r_dim
        self.num_bits = num_bits
        self.temperature = torch.Tensor([100])

        # Networks
        self.group_enc = GroupEncoder(in_dim=x_dim+r_dim, out_dim=u_dim)
        self.inst_enc = InstEncoder(in_dim=x_dim+r_dim+u_dim, out_dim=v_dim)
        self.decoder = Decoder(in_dim=u_dim+v_dim+r_dim, out_dim=x_dim)

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
        # Create new dimension for scales and concatenate positional embedding
        x = x.unsqueeze(3)
        x = x.expand(-1, -1, -1, self.num_bits, -1)
        x = bin_pos_emb(x)

        # Sample q(u|{x})
        u_loc, u_scale = self.group_enc.forward(x)
        qu = Normal(u_loc, u_scale)
        u = qu.rsample()

        # Sample q(v|x,u)
        v = self.inst_enc.forward(x, u)
        qv = RelaxedOneHotCategorical(self.temperature, logits=v)
        # v = qv.rsample()

        # Turn vector of weights into one hot using hard max
        _, idcs = v.max(dim=-1)
        idcs = idcs.unsqueeze(-1).broadcast_to(v.shape)
        v_hard = torch.zeros_like(v)
        v_hard.scatter_(-1, idcs, 1)

        # Straight-Through trick
        v_hard = (v_hard - v).detach() + v

        return qu, qv, u, v

    def generate(self, u, v):
        # Create new dimension for scales and concatenate positional embedding
        v = bin_pos_emb(v)

        # Generative data dist
        x_loc = self.decoder.forward(u, v)

        return x_loc

    # ELBO loss for hierarchical variational autoencoder
    def elbo_func(self, x):
        # Latent inference
        qu, qv, u, v = self.inference(x)
        u = torch.zeros_like(u)

        # Latent prior
        pu, _ = self.prior(u, v)

        # Generative data dist
        x_loc = self.generate(u, v)

        # Losses
        kl_u = torch.mean(qu.log_prob(u) - pu.log_prob(u))
        kl_v = 0
        lik = torch.mean(torch.abs(x - x_loc))
        return lik

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        # sample q(u,{v}|{x})
        _, _, u, v = self.inference(x)
        # decode p({x}|u,{v})
        x_loc = self.generate(u, v)
        return x_loc

    # define a helper function for translating images
    def translate(self, x, y):
        # sample q(u,{v}|{x})
        _, _, _, v = self.inference(x)
        # sample q(uy|{y})
        _, _, u, _ = self.inference(y)
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
