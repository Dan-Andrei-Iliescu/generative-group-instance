import torch
import torch.nn as nn


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


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(u_i|{x}_i)
class Discriminator(nn.Module):
    def __init__(self, x_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # define the forward computation on the data x
        hidden = self.softplus(self.fc1(x))
        # aggregate embeddings
        hidden = torch.mean(hidden, dim=0, keepdim=True)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        p = self.sigmoid(self.fc2(hidden))
        return p
