import torch
import numpy as np
from utils.plots import plot_1D_rec, plot_1D_trans, plot_1D_latent


# turn input numpy array into torch tensor
def prepare_data(model, x):
    x = torch.Tensor(x)
    # if on GPU put mini-batch into CUDA memory
    if model.cuda:
        x = x.cuda()
    return x


def trans_test(model, test_x, test_y, test_trans, epoch):
    error = 0
    for x, y, gt in zip(test_x, test_y, test_trans):
        trans = model.translate(
            prepare_data(model, x), prepare_data(model, y))\
            .detach().cpu().numpy()
        error += np.sum((trans - gt)**2)
        plot_1D_trans(x, y, trans, f"Translation_{epoch}")
    print("[epoch %03d]  Translation error: %.4f" % (epoch, error))


def rec_test(model, test_data, epoch):
    error = 0
    rec_data = []
    for x in test_data:
        rec = model.reconstruct(
            prepare_data(model, x)).detach().cpu().numpy()
        error += np.sum((rec - x)**2)
        rec_data.append(rec)
    print("[epoch %03d]  Reconstruction error: %.4f" % (epoch, error))
    plot_1D_rec(test_data, rec_data, f"Reconstruction_{epoch}")


def latent_test(model, test_data, epoch):
    v_list = []
    for x in test_data:
        _, v = model.encode(prepare_data(model, x))
        v_list.append(v.detach().cpu().numpy())
    plot_1D_latent(v_list, f"Latent_{epoch}")
