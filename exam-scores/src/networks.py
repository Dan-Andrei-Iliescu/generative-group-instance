import torch
import torch.nn as nn


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z_i|x_i)
class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc41 = nn.Linear(h_dim, z_dim)
        self.fc42 = nn.Linear(h_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.Softplus()
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the data x
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        hidden = self.relu(self.fc3(hidden))
        v_loc = self.fc41(hidden)
        v_scale = self.softplus(self.fc42(hidden)) + 1e-8
        return v_loc, v_scale


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(u_i|{x}_i)
class GroupEncoder(nn.Module):
    def __init__(self, x_dim, u_dim, h_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc41 = nn.Linear(h_dim, u_dim)
        self.fc42 = nn.Linear(h_dim, u_dim)
        # setup the non-linearities
        self.relu = nn.Softplus()
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the data x
        hidden = self.relu(self.fc1(x))
        hidden = self.fc2(hidden)
        # aggregate embeddings
        hidden = torch.sum(hidden, dim=1, keepdim=True)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        hidden = self.relu(self.fc3(hidden))
        u_loc = self.fc41(hidden)
        u_scale = self.softplus(self.fc42(hidden)) + 1e-8
        return u_loc, u_scale


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(v_ij|x_ij, u_i)
class InstEncoder(nn.Module):
    def __init__(self, x_dim, u_dim, v_dim, h_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim+u_dim, h_dim+u_dim)
        self.fc41 = nn.Linear(h_dim+u_dim, v_dim)
        self.fc42 = nn.Linear(h_dim+u_dim, v_dim)
        # setup the non-linearities
        self.relu = nn.Softplus()
        self.softplus = nn.Softplus()

    def forward(self, x, u):
        # define the forward computation on the data x
        hidden = self.relu(self.fc1(x))
        hidden = self.fc2(hidden)
        # concatenate u with embedding of x
        u = torch.broadcast_to(u, [-1, hidden.shape[-2], u.shape[-1]])
        hidden = torch.cat([hidden, u], dim=-1)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        hidden = self.relu(self.fc3(hidden))
        v_loc = self.fc41(hidden)
        v_scale = self.softplus(self.fc42(hidden)) + 1e-8
        return v_loc, v_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(3*x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()

    def forward(self, u, v):
        # concatenate u with v
        u = torch.broadcast_to(u, [-1, v.shape[1], u.shape[2]])
        z = torch.cat([u, v], dim=2)
        hidden = self.relu(self.fc1(z))
        hidden = self.relu(self.fc2(hidden))
        hidden = self.relu(self.fc3(hidden))
        x_loc = self.fc4(hidden)
        return x_loc


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class DecoderGiven(nn.Module):
    def __init__(self, xy_ratio, x_dim):
        super().__init__()
        self.xy_ratio = xy_ratio
        self.x_dim = x_dim

    def forward(self, u, v):
        mult = 1
        batch_size = u.shape[0]
        u = torch.broadcast_to(u, [batch_size, v.shape[1], u.shape[2]])
        x = torch.zeros_like(v)
        x[:, :, 0] = self.xy_ratio * mult * u[:, :, 0] + \
            (u[:, :, 1] / 3 + 1)**self.xy_ratio * v[:, :, 0]
        x[:, :, 1] = mult * u[:, :, 2] + self.xy_ratio * \
            (u[:, :, 3] / 3 + 1) * v[:, :, 1]
        return x


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(u_i|{x}_i)
class Discriminator(nn.Module):
    def __init__(self, x_dim, h_dim=128):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, 1)
        # setup the non-linearities
        self.relu = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the data x
        hidden = self.relu(self.fc1(x))
        hidden = self.fc2(hidden)
        # aggregate embeddings
        hidden = torch.sum(hidden, dim=-2, keepdim=True)
        hidden = self.relu(self.fc3(hidden))
        d = self.fc4(hidden)
        return d
