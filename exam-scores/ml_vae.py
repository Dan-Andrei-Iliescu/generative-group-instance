import random

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

import numpy as np
import fire

from utils.plots import plot_1D_rec, plot_1D_trans, plot_1D_latent
from toy_data import generate_dataset


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(u_i|{x}_i)
class GroupEncoder(nn.Module):
    def __init__(self, x_dim, u_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, u_dim)
        self.fc22 = nn.Linear(hidden_dim, u_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the data x
        hidden = self.softplus(self.fc1(x))
        # aggregate embeddings
        hidden = torch.mean(hidden, dim=0, keepdim=True)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        u_loc = self.fc21(hidden)
        u_scale = self.softplus(self.fc22(hidden))
        return u_loc, u_scale


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(v_ij|x_ij, u_i)
class InstEncoder(nn.Module):
    def __init__(self, x_dim, u_dim, v_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim+u_dim, v_dim)
        self.fc22 = nn.Linear(hidden_dim+u_dim, v_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x, u):
        # define the forward computation on the data x
        hidden = self.softplus(self.fc1(x))
        # concatenate u with embedding of x
        u = torch.broadcast_to(u, [hidden.shape[0], u.shape[1]])
        hidden = torch.cat([hidden, u], dim=-1)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        v_loc = self.fc21(hidden)
        v_scale = self.softplus(self.fc22(hidden))
        return v_loc, v_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, x_dim, u_dim, v_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(u_dim+v_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, x_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, u, v):
        # concatenate u with v
        u = torch.broadcast_to(u, [v.shape[0], u.shape[1]])
        z = torch.cat([u, v], dim=-1)
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        x_loc = self.fc2(hidden)
        return x_loc


# define a PyTorch module for the VAE
class VAE(nn.Module):
    def __init__(
            self, x_dim=1, u_dim=2, v_dim=1, h_dim=32, lr=1e-3, cuda=False):
        super().__init__()
        self.u_dim = u_dim
        self.v_dim = v_dim

        # create the encoder and decoder networks
        self.group_enc = GroupEncoder(x_dim, u_dim, h_dim)
        self.inst_enc = InstEncoder(x_dim, u_dim, v_dim, h_dim)
        self.decoder = Decoder(x_dim, u_dim, v_dim, h_dim)

        # calling cuda() here will put all the parameters of
        # the encoder and decoder networks into gpu memory
        self.cuda = cuda
        if self.cuda:
            self.cuda()

        # setup the optimizer
        model_params = list(self.group_enc.parameters()) + \
            list(self.inst_enc.parameters()) + \
            list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(
            model_params, lr=lr, betas=(0.5, 0.999))

    # define the generative model p(u) \prod_i p(v_i) p(x_i|u,v_i)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # sample p(u)
        u_loc = torch.zeros(1, self.u_dim, dtype=x.dtype, device=x.device)
        u_scale = torch.ones(1, self.u_dim, dtype=x.dtype, device=x.device)
        u = pyro.sample("group", dist.Normal(u_loc, u_scale).to_event(1))

        with pyro.plate("data", x.shape[0]):
            # sample p(v)
            v_loc = torch.zeros(
                x.shape[0], self.v_dim, dtype=x.dtype, device=x.device)
            v_scale = torch.ones(
                x.shape[0], self.v_dim, dtype=x.dtype, device=x.device)
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

        with pyro.plate("data", x.shape[0]):
            # sample q(v|x,u)
            v_loc, v_scale = self.inst_enc.forward(x, u)
            v = pyro.sample("inst", dist.Normal(v_loc, v_scale).to_event(1))

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

    # turn input numpy array into torch tensor
    def prepare_data(self, x):
        x = torch.Tensor(x)
        # if on GPU put mini-batch into CUDA memory
        if self.cuda:
            x = x.cuda()
        return x

    # plot translation
    def trans_test(self, test_data, epoch):
        perm = test_data.copy()
        random.seed(100)
        random.shuffle(perm)
        idx = 0
        for x, y in zip(test_data, perm):
            trans = self.translate(
                self.prepare_data(x), self.prepare_data(y))\
                .detach().cpu().numpy()
            plot_1D_trans(
                x, y, trans, "Hierarchical VAE", f"hvae/trans_{idx}")
            idx += 1

    # plot reconstruction
    def rec_test(self, test_data, epoch):
        rec_data = []
        for x in test_data:
            rec_data.append(self.reconstruct(
                self.prepare_data(x)).detach().cpu().numpy())
        plot_1D_rec(
            test_data, rec_data, "Hierarchical VAE", f"hvae/rec_{epoch}")

    # plot latent
    def latent_test(self, test_data, epoch):
        v_list = []
        for x in test_data:
            _, v = self.encode(self.prepare_data(x))
            v_list.append(v.detach().cpu().numpy())
        plot_1D_latent(
            v_list, "Hierarchical VAE", f"hvae/latent_{epoch}")


def main(
        x_dim=1, num_epochs=10, test_freq=3, lr=1e-3, cuda=False,
        num_train_groups=8000, num_test_groups=6):
    # clear param store
    pyro.clear_param_store()

    # setup data lists
    train_data, test_data = generate_dataset(
        x_dim=x_dim, num_train_groups=num_train_groups,
        num_test_groups=num_test_groups)

    # setup the VAE
    vae = VAE(x_dim=x_dim, cuda=cuda, lr=lr)

    # training loop
    for epoch in range(num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x in train_data:
            # do ELBO gradient and accumulate loss
            epoch_loss += vae.step(vae.prepare_data(x))

        # report training diagnostics
        normalizer_train = len(train_data)
        total_epoch_loss_train = epoch_loss / normalizer_train
        print("[epoch %03d]  average training loss: %.4f" %
              (epoch, total_epoch_loss_train))

        if epoch % test_freq == 0:
            # plot reconstruction
            vae.rec_test(test_data, epoch)
            # plot translation
            vae.trans_test(test_data, epoch)
            # plot latents
            vae.latent_test(test_data, epoch)

            # initialize loss accumulator
            test_error = 0.
            # compute the loss over the entire test set
            for x in test_data:
                # compute ELBO estimate and accumulate loss
                rec_x = vae.reconstruct(vae.prepare_data(x))\
                    .detach().cpu().numpy()
                test_error += np.sum((x - rec_x)**2)

            # report test diagnostics
            normalizer_test = len(test_data)
            total_epoch_loss_test = test_error / normalizer_test
            print("[epoch %03d]  average test error: %.4f" %
                  (epoch, total_epoch_loss_test))

    return vae


if __name__ == '__main__':
    fire.Fire(main)
