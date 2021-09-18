import torch

from src.ml_vae import Model as Super
from src.networks import Discriminator


# define a PyTorch module for the VAE
class Model(Super):
    def __init__(
            self, x_dim=1, u_dim=2, v_dim=1, h_dim=32, lr=1e-3, cuda=False):
        super().__init__(
            x_dim=x_dim, u_dim=u_dim, v_dim=v_dim, h_dim=h_dim, lr=lr,
            cuda=cuda)
        self.name = "v_vs_n_gp"

        # create the discriminator net
        self.disc = Discriminator(x_dim, h_dim)
        self.disc_coeff = 10000

        # setup the disc optimizer
        self.optim_disc = torch.optim.Adam(
            self.disc.parameters(), lr=lr, betas=(0.5, 0.999))

    def grad_penalty(self, loss):
        # Creates gradients
        grad_params = torch.autograd.grad(outputs=loss,
                                          inputs=self.disc.parameters(),
                                          create_graph=True)

        # Computes the penalty term and adds it to the loss
        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return grad_norm

    def disc_loss(self, x):
        # Get instance variables
        _, v = self.guide(x)

        # Disc loss for instance variables
        fake_d = self.disc(v)

        # Disc loss for normal distribution
        gt = torch.normal(mean=torch.zeros_like(v), std=torch.ones_like(v))
        real_d = self.disc(gt)

        return real_d - fake_d

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
