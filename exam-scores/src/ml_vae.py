import sys
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from src.networks import GroupEncoder, InstEncoder, Decoder


# define a PyTorch module for the VAE
class Model(nn.Module):
    def __init__(
            self, x_dim=1, u_dim=2, v_dim=1, h_dim=32, lr=1e-3, cuda=False):
        super().__init__()
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.name = "ml_vae"

        # create the encoder and decoder networks
        self.group_enc = GroupEncoder(x_dim, u_dim, h_dim)
        self.inst_enc = InstEncoder(x_dim, u_dim, v_dim, h_dim)
        self.decoder = Decoder(x_dim, u_dim, v_dim, h_dim)

        # calling cuda() here will put all the parameters of
        # the encoder and decoder networks into gpu memory
        self.use_cuda = cuda
        if self.use_cuda:
            self.cuda()

        # setup the optimizer
        self.model_params = list(self.group_enc.parameters()) + \
            list(self.inst_enc.parameters()) + \
            list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(
            self.model_params, lr=lr, betas=(0.5, 0.999))

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
        u_loc, u_scale = self.group_enc.forward(x)
        u = pyro.sample("group", dist.Normal(u_loc, u_scale).to_event(1))

        # sample q(v|x,u)
        v_loc, v_scale = self.inst_enc.forward(x, u)
        v = pyro.sample("inst", dist.Normal(
            v_loc, v_scale).to_event(1))

        # return the latents
        return u, v

    # ELBO loss for hierarchical variational autoencoder
    def elbo_loss(self, x):
        # run the guide and trace its execution
        guide_trace = poutine.trace(self.guide).get_trace(x)
        # run the model and replay it against the samples from the guide
        model_trace = poutine.trace(
            poutine.replay(self.model, trace=guide_trace)).get_trace(x)
        # construct the elbo loss function
        return -1 * (model_trace.log_prob_sum() - guide_trace.log_prob_sum())

    # one training step
    def step(self, x):
        self.optimizer.zero_grad()
        elbo_loss = self.elbo_loss(x)
        loss = elbo_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    # define a helper function for encoding images
    def encode(self, x):
        # sample q(u|{x})
        u_loc, u_scale = self.group_enc(x)
        u = dist.Normal(u_loc, u_scale).sample()
        # sample q({v}|{x},u)
        v_loc, v_scale = self.inst_enc(x, u)
        v = dist.Normal(v_loc, v_scale).sample()
        return u, v

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        # sample q(u,{v}|{x})
        u, v = self.encode(x)
        # decode p({x}|u,{v})
        x_loc = self.decoder(u, v)
        return x_loc

    # define a helper function for translating images
    def translate(self, x, y):
        # sample q(u,{v}|{x})
        _, v = self.encode(x)
        # sample q(uy|{y})
        u_loc, u_scale = self.group_enc(y)
        u = dist.Normal(u_loc, u_scale).sample()
        # decode p({x}|uy,{v})
        trans = self.decoder(u, v)
        return trans
