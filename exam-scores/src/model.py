import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist

from src.networks import Encoder, GroupEncoder, InstEncoder, Decoder
from src.loss_funcs import elbo_func, v_vs_n_func, grad_penalty_func


# define a PyTorch module for the VAE
class Model(nn.Module):
    def __init__(
            self, group_acc=None, inst_cond=True, reg=None,
            x_dim=1, u_dim=2, v_dim=1, h_dim=32, lr=1e-4, cuda=False):
        super().__init__()
        self.group_acc = group_acc
        self.inst_cond = inst_cond
        self.reg = reg
        self.u_dim = u_dim
        self.v_dim = v_dim

        # create group encoder network depending on the accumulation method
        if self.group_acc == "acc":
            self.group_enc = Encoder(x_dim, u_dim, h_dim)
        else:
            self.group_enc = GroupEncoder(x_dim, u_dim, h_dim)

        # create instance encoder network depending on whether
        # the instance encoder takes as input the group variable
        if self.inst_cond is False:
            self.inst_enc = Encoder(x_dim, z_dim=v_dim, h_dim=h_dim)
        else:
            self.inst_enc = InstEncoder(x_dim, u_dim, v_dim, h_dim)

        # create decoder network
        self.decoder = Decoder(x_dim, u_dim, v_dim, h_dim)

        # create adversary network depending on which regularization to use
        if self.reg == "v_vs_n":
            self.adv = GroupEncoder(x_dim=v_dim, u_dim=1, h_dim=h_dim)
            self.adv_func = v_vs_n_func
        """
        elif self.reg == "vx_vs_v_x":
            self.adv = Encoder(x_dim=x_dim+v_dim, z_dim=1, h_dim=h_dim)
        """
        self.adv_coeff = 10000

        # calling cuda() here will put all the parameters of the model on GPU
        self.use_cuda = cuda
        if self.use_cuda:
            self.cuda()

        # setup the optimizer
        self.model_params = list(self.group_enc.parameters()) + \
            list(self.inst_enc.parameters()) + \
            list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(
            self.model_params, lr=lr, betas=(0.1, 0.999),
            weight_decay=1e-2, eps=0.1)

        # Optimizer for regularization
        if self.reg is not None:
            # setup the disc optimizer
            self.optim_adv = torch.optim.Adam(
                self.adv.parameters(), lr=lr, betas=(0.1, 0.999),
                weight_decay=1e-2, eps=0.1)

    # define the generative model p(u) \prod_i p(v_i) p(x_i|u,v_i)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # sample p(u)
        u_loc = torch.zeros(1, self.u_dim, dtype=x.dtype, device=x.device)
        u_scale = torch.ones(1, self.u_dim, dtype=x.dtype, device=x.device)
        u = pyro.sample("group", dist.Normal(u_loc, u_scale).to_event(1))

        # sample p(v)
        v_loc = torch.zeros(
            x.shape[0], x.shape[1], self.v_dim, dtype=x.dtype, device=x.device)
        v_scale = torch.ones(
            x.shape[0], x.shape[1], self.v_dim, dtype=x.dtype, device=x.device)
        v = pyro.sample("inst", dist.Normal(v_loc, v_scale).to_event(1))

        # sample p(x|u,v)
        x_loc = self.decoder.forward(u, v)
        x_scale = 0.1
        pyro.sample(
            "x", dist.Normal(x_loc, x_scale).to_event(1), obs=x)

        # return the loc so we can visualize it later
        return x_loc

    # define the variational posterior q(u|{x}) \prod_i q(v_i|x_i,u)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("group_encoder", self.group_enc)
        pyro.module("instance_encoder", self.inst_enc)

        # sample q(u|{x})
        if self.group_acc == "acc":
            u_loc_raw, u_scale_raw = self.group_enc.forward(x)
            u_scale = torch.mean(u_scale_raw, dim=1, keepdim=True)
            u_loc = torch.mean(u_loc_raw, dim=1, keepdim=True)
        else:
            u_loc, u_scale = self.group_enc.forward(x)
        if torch.isnan(torch.sum(u_loc)):
            print(
                "!!! u_loc is NaN for " +
                f"{self.group_acc}-{self.inst_cond}-{self.reg}")
        if torch.isnan(torch.sum(u_scale)):
            print(
                "!!! u_scale is NaN for " +
                f"{self.group_acc}-{self.inst_cond}-{self.reg}")
        u = pyro.sample("group", dist.Normal(u_loc, u_scale).to_event(1))

        # sample q(v|x,u)
        if not self.inst_cond:
            v_loc, v_scale = self.inst_enc.forward(x)
        else:
            v_loc, v_scale = self.inst_enc.forward(x, u)
        if torch.isnan(torch.sum(v_loc)):
            print(
                "!!! v_loc is NaN for " +
                f"{self.group_acc}-{self.inst_cond}-{self.reg}")
        if torch.isnan(torch.sum(v_scale)):
            print(
                "!!! v_scale is NaN for " +
                f"{self.group_acc}-{self.inst_cond}-{self.reg}")
        v = pyro.sample("inst", dist.Normal(
            v_loc, v_scale).to_event(1))

        # return the latents
        return u, v

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        # sample q(u,{v}|{x})
        u, v = self.guide(x)
        # decode p({x}|u,{v})
        x_loc = self.decoder(u, v)
        return x_loc

    # define a helper function for translating images
    def translate(self, x, y):
        # sample q(u,{v}|{x})
        _, v = self.guide(x)
        # sample q(uy|{y})
        u, _ = self.guide(y)
        # decode p({x}|uy,{v})
        trans = self.decoder(u, v)
        return trans

    # one training step
    def step(self, x):
        # adversarial loss
        if self.reg is not None:
            self.optim_adv.zero_grad()
            loss = self.adv_coeff * self.adv_func(self, x)
            loss += grad_penalty_func(self.adv.parameters(), loss)
            loss.backward()
            nn.utils.clip_grad_value_(self.adv.parameters(), clip_value=1.)
            self.optim_adv.step()
            self.optim_adv.zero_grad()

        # model loss
        self.optimizer.zero_grad()
        loss = 0
        if self.reg is not None:
            loss -= self.adv_coeff * self.adv_func(self, x)
            loss += grad_penalty_func(self.adv.parameters(), loss)
        loss += elbo_func(self, x)
        loss += grad_penalty_func(self.model_params, loss)
        loss.backward()
        nn.utils.clip_grad_value_(self.model_params, clip_value=1.)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss
