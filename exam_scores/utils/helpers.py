import random
import torch
from utils.plots import plot_1D_rec, plot_1D_trans, plot_1D_latent


# turn input numpy array into torch tensor
def prepare_data(model, x):
    x = torch.Tensor(x)
    # if on GPU put mini-batch into CUDA memory
    if model.cuda:
        x = x.cuda()
    return x


# plot translation
def trans_test(model, test_data, epoch):
    perm = test_data.copy()
    random.seed(100)
    random.shuffle(perm)
    idx = 0
    for x, y in zip(test_data, perm):
        trans = model.translate(
            prepare_data(model, x), prepare_data(model, y))\
            .detach().cpu().numpy()
        plot_1D_trans(
            x, y, trans, "Hierarchical VAE", f"hvae/trans_{idx}")
        idx += 1


# plot reconstruction
def rec_test(model, test_data, epoch):
    rec_data = []
    for x in test_data:
        rec_data.append(model.reconstruct(
            prepare_data(model, x)).detach().cpu().numpy())
    plot_1D_rec(
        test_data, rec_data, "Hierarchical VAE", f"hvae/rec_{epoch}")


# plot latent
def latent_test(model, test_data, epoch):
    v_list = []
    for x in test_data:
        _, v = model.encode(prepare_data(model, x))
        v_list.append(v.detach().cpu().numpy())
    plot_1D_latent(
        v_list, "Hierarchical VAE", f"hvae/latent_{epoch}")
