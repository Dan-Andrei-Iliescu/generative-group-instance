import torch
import torch.nn as nn

from src.networks import Encoder, InstEncoder, Decoder
from src.loss_funcs import elbo_func


# define a PyTorch module for the VAE
class Model(nn.Module):
    def __init__(
            self, inst_cond=True, x_dim=2, h_dim=32, xy_ratio=1., lr=1e-4,
            wd=1e-0):
        super().__init__()
        self.inst_cond = inst_cond

        self.group_enc = Encoder(x_dim, 2*x_dim, h_dim)

        # create instance encoder network depending on whether
        # the instance encoder takes as input the group variable
        if not self.inst_cond:
            self.inst_enc = Encoder(x_dim, z_dim=x_dim, h_dim=h_dim)
        else:
            self.inst_enc = InstEncoder(x_dim, 2*x_dim, x_dim, h_dim)

        # create decoder network
        self.decoder = Decoder(x_dim=x_dim, h_dim=h_dim)

        # setup the optimizer
        self.model_params = list(self.group_enc.parameters()) + \
            list(self.inst_enc.parameters()) + \
            list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(
            self.model_params, lr=lr, weight_decay=wd)

    # define the generative prior p(u) \prod_i p(v_i)
    def prior(self, u, v, sample=True):
        pu = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u))
        u = pu.rsample()
        pv = torch.distributions.Normal(
            torch.zeros_like(v), torch.ones_like(v))
        v = pv.rsample()

        if sample:
            return u, v
        else:
            return pu, pv

    # define the variational posterior q(u|{x}) \prod_i q(v_i|x_i,u)
    def inference(self, x, sample=True):
        # sample q(u|{x})
        u_loc, u_scale = self.group_enc.forward(x)
        u_scale = torch.mean(u_scale, dim=1, keepdim=True)
        u_loc = torch.mean(u_loc, dim=1, keepdim=True)
        if torch.isnan(torch.sum(u_loc)):
            print(
                "!!! u_loc is NaN for " +
                f"{self.group_acc}-{self.inst_cond}-{self.reg}")
        if torch.isnan(torch.sum(u_scale)):
            print(
                "!!! u_scale is NaN for " +
                f"{self.group_acc}-{self.inst_cond}-{self.reg}")
        qu = torch.distributions.Normal(u_loc, u_scale)
        u = qu.rsample()

        # sample q(v|x,u)
        if not self.inst_cond:
            v_loc, v_scale = self.inst_enc.forward(x)
        else:
            v_loc, v_scale = self.inst_enc.forward(x, u)
        """
        if torch.isnan(torch.sum(v_loc)):
            print(
                "!!! v_loc is NaN for " +
                f"{self.group_acc}-{self.inst_cond}-{self.reg}")
        if torch.isnan(torch.sum(v_scale)):
            print(
                "!!! v_scale is NaN for " +
                f"{self.group_acc}-{self.inst_cond}-{self.reg}")
        """
        qv = torch.distributions.Normal(v_loc, v_scale)
        v = qv.rsample()

        # return the latents
        if sample:
            return u, v
        else:
            return qu, qv

    # define a helper function for computing the adversary prediction
    def nemeth(self, x, v):
        adv_v, _ = self.adv_v(x, v)
        adv_v = torch.tanh(adv_v)
        return adv_v

    # define a helper function for computing the adversary prediction
    def ours(self, x, v):
        adv_u, _ = self.adv_u(x)
        adv_v, _ = self.adv_v(v, adv_u)
        adv_v = torch.tanh(adv_v)
        return adv_v

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        # sample q(u,{v}|{x})
        u, v = self.inference(x)
        # decode p({x}|u,{v})
        x_loc = self.decoder(u, v)
        return x_loc

    # define a helper function for translating images
    def translate(self, x, y):
        # sample q(u,{v}|{x})
        _, v = self.inference(x)
        # sample q(uy|{y})
        u, _ = self.inference(y)
        # decode p({x}|uy,{v})
        trans = self.decoder(u, v)
        return trans

    # one training step
    def step(self, x):
        # model loss
        self.optimizer.zero_grad()
        loss = elbo_func(self, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
