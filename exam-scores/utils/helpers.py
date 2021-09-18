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
    ok = True
    for x, y, gt in zip(test_x, test_y, test_trans):
        trans = model.translate(
            prepare_data(model, x), prepare_data(model, y))\
            .detach().cpu().numpy()
        error += np.sum((trans - gt)**2)

        if ok:
            plot_1D_trans(x, y, trans, f"Translation_{model.name}_{epoch}")
            ok = False
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
    plot_1D_rec(test_data, rec_data, f"Reconstruction_{model.name}_{epoch}")


def latent_test(model, test_data, epoch):
    v_list = []

    for x in test_data:
        _, v = model.encode(prepare_data(model, x))
        v = v.detach().cpu().numpy()
        v_list.append(v)

        # Error
        means = np.mean(v, axis=0, keepdims=True)
        mean_error = np.sum(means**2)
        vars = np.mean((v - means)**2, axis=0, keepdims=True)
        var_error = np.sum((vars - 1)**2)
    print("[epoch %03d]  latent mean error: %.4f, latent var error: %.4f" %
          (epoch, mean_error, var_error))
    plot_1D_latent(v_list, f"Latent_{model.name}_{epoch}")
