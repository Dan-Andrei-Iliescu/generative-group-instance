import torch
import torch.nn as nn

from src.networks import Encoder, InstEncoder, Decoder


# define a PyTorch module for the VAE
class Model(nn.Module):
    def __init__(self, cond=False, z_dim=4, h_dim=16):
        super().__init__()
        self.cond = cond

        # create group encoder network depending on the accumulation method
        self.u_enc = Encoder(z_dim, h_dim)

        # create instance encoder network depending on whether
        # the instance encoder takes as input the group variable
        if self.cond:
            self.v_enc = InstEncoder(z_dim, h_dim)
        else:
            self.v_enc = Encoder(z_dim, h_dim)

        # create decoder network
        self.decoder = Decoder()

        # setup the optimizer
        self.params = list(self.u_enc.parameters()) + \
            list(self.v_enc.parameters()) + \
            list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=1e-3)

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
        u_loc, u_scale = self.u_enc.forward(x)
        u_scale = torch.mean(u_scale, dim=0, keepdim=True)
        u_loc = torch.mean(u_loc, dim=0, keepdim=True)
        qu = torch.distributions.Normal(u_loc, u_scale)
        u = qu.rsample()

        # sample q(v|x,u)
        if self.cond:
            v_loc, v_scale = self.v_enc.forward(x, u)
        else:
            v_loc, v_scale = self.v_enc.forward(x)
        qv = torch.distributions.Normal(v_loc, v_scale)
        v = qv.rsample()

        # return the latents
        if sample:
            return u, v
        else:
            return qu, qv

    def prep(self, x):
        return torch.tensor(x, dtype=torch.float32)

    def unprep(self, x):
        return x.cpu().detach().numpy()

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        # sample q(u,{v}|{x})
        u, v = self.inference(self.prep(x))
        # decode p({x}|u,{v})
        x_rec = self.decoder(u, v)
        return self.unprep(x_rec)

    # define a helper function for translating images
    def translate(self, x, y):
        # sample q(u,{v}|{x})
        _, v = self.inference(x)
        # sample q(uy|{y})
        u, _ = self.inference(y)
        # decode p({x}|uy,{v})
        trans = self.decoder(u, v)
        return trans

    def decode(self, u, v):
        return self.unprep(self.decoder(self.prep(u), self.prep(v)))

    # ELBO loss for hierarchical variational autoencoder
    def elbo_func(self, x):
        # Latent inference dists
        qu, qv = self.inference(x, sample=False)

        # Sample from latent inference dists
        u = qu.rsample()
        v = qv.rsample()

        # Latent prior dists
        pu, pv = self.prior(u, v, sample=False)

        # Generative data dist
        x_loc = self.decoder.forward(u, v)

        # Losses
        kl_u = torch.mean(qu.log_prob(u) - pu.log_prob(u))
        # print(f"KL u {kl_u}")
        kl_v = torch.mean(qv.log_prob(v) - pv.log_prob(v))
        # print(f"KL v {kl_v}")
        lik = torch.mean(torch.abs(x - x_loc))
        # print(f"lik {lik}")
        # loss = kl_u + kl_v + lik
        loss = lik

        # print(f"ELBO loss {loss}")
        return loss

    # one training step
    def step(self, x):
        self.optimizer.zero_grad()
        loss = self.elbo_func(self.prep(x))
        loss.backward()
        self.optimizer.step()

    def get_latents(self, x):
        u, v = self.inference(self.prep(x))
        return self.unprep(u), self.unprep(v)
