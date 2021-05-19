# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import torch
import torch.nn as nn

from toy_data import generate_dataset

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

import matplotlib.pyplot as plt


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        # self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(x_dim, z_dim)
        self.fc22 = nn.Linear(x_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the data x
        # compute the hidden units
        # hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(x)
        z_scale = torch.exp(self.fc22(x))
        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        # self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(z_dim, x_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        # hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        x_loc = self.fc2(z)
        return x_loc


# define a PyTorch module for the VAE
class VAE(nn.Module):
    def __init__(self, x_dim=1, z_dim=3, hidden_dim=16, use_cuda=False):
        super().__init__()
        self.x_dim = x_dim
        # create the encoder and decoder networks
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(x_dim, z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            x_loc = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs", dist.Normal(x_loc, 0.1).to_event(1), obs=x)
            # return the loc so we can visualize it later
            return x_loc

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        x_loc = self.decoder(z)
        return x_loc

    def plot_rec_error(self, x):
        x_rec = self.reconstruct_img(x).detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        plt.scatter(x[:, 0], x[:, 1], color='orange')
        plt.scatter(x_rec[:, 0], x_rec[:, 1], color='blue')
        """
        for idx in range(x.shape[0]):
            plt.plot(x[idx, :2], x_rec[idx, :2], color='blue')
        """


def main(args):
    # clear param store
    pyro.clear_param_store()

    # setup data lists
    train_data, test_data = generate_dataset(
        x_dim=args.x_dim, num_train_groups=1000, train_lam=8,
        num_test_groups=100, test_lam=9)

    # setup the VAE
    vae = VAE(x_dim=args.x_dim, use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x in train_data:
            x = torch.Tensor(x)
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        # report training diagnostics
        normalizer_train = len(train_data)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for x, group_idx in zip(test_data, range(len(test_data))):
                x = torch.Tensor(x)
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

                if group_idx < 4:
                    vae.plot_rec_error(x)
            plt.show()

            # report test diagnostics
            normalizer_test = len(test_data)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

    return vae


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.6.0')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-x', '--x_dim', default=2, type=int)
    parser.add_argument('-n', '--num-epochs', default=31, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=10, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-i-tsne', '--tsne_iter', default=3, type=int, help='epoch when tsne visualization runs')
    args = parser.parse_args()

    model = main(args)