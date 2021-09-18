import torch
import torch.nn as nn

from src.ml_vae import Model as Super
from src.networks import Discriminator


# define a PyTorch module for the VAE
class Model(Super):
    def __init__(
            self, x_dim=1, u_dim=2, v_dim=1, h_dim=32, lr=1e-3, cuda=False):
        super().__init__(
            x_dim=x_dim, u_dim=u_dim, v_dim=v_dim, h_dim=h_dim, lr=lr,
            cuda=cuda)
        self.name = "v_vs_n"

        # create the discriminator net
        self.disc = Discriminator(x_dim)
        self.disc_coeff = 10000

        # setup the disc optimizer
        self.optim_disc = torch.optim.Adam(
            self.disc.parameters(), lr=lr, betas=(0.5, 0.999))

    def disc_loss(self, x):
        # Loss criterion
        criterion = nn.BCEWithLogitsLoss()

        # Get instance variables
        _, v = self.guide(x)

        # Disc loss for instance variables
        fake_d = self.disc(v)
        fake_label = torch.zeros_like(fake_d)
        fake_loss = criterion(fake_d, fake_label)

        # Disc loss for normal distribution
        gt = torch.normal(mean=torch.zeros_like(v), std=torch.ones_like(v))
        real_d = self.disc(gt)
        real_label = torch.ones_like(real_d)
        real_loss = criterion(real_d, real_label)

        return real_loss + fake_loss

    # one training step
    def step(self, x):
        self.optim_disc.zero_grad()
        loss = self.disc_coeff * self.disc_loss(x)
        loss.backward(create_graph=True)
        self.optim_disc.step()
        self.optim_disc.zero_grad()

        self.optimizer.zero_grad()
        elbo_loss = self.elbo_loss(x)
        disc_loss = self.disc_coeff * self.disc_loss(x)
        loss = elbo_loss - disc_loss
        # loss = - disc_loss
        loss.backward(create_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss
