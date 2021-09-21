import torch
import numpy as np


# turn input numpy array into torch tensor
def prepare_data(model, x):
    x = torch.Tensor(x)
    # if on GPU put mini-batch into CUDA memory
    if model.cuda:
        x = x.cuda()
    return x


def rec_error(test_x, x_rec):
    error = []
    for x, rec in zip(test_x, x_rec):
        error.append(np.mean((x - rec)**2))
    return error


def latent_error(v_list):
    mean_error = []
    var_error = []
    for v in v_list:
        means = np.mean(v, axis=0, keepdims=True)
        mean_error.append(np.mean(means**2))
        vars = np.mean((v - means)**2, axis=0, keepdims=True)
        var_error.append(np.mean((vars - 1)**2))
    return mean_error, var_error


def trans_test(model, test_x, test_y):
    trans = []
    for x, y in zip(test_x, test_y):
        trans.append(model.translate(
            prepare_data(model, x), prepare_data(model, y))
            .detach().cpu().numpy())
    return trans


def rec_test(model, test_data):
    rec = []
    for x in test_data:
        rec.append(model.reconstruct(
            prepare_data(model, x)).detach().cpu().numpy())
    return rec


def latent_test(model, test_data):
    v_list = []
    for x in test_data:
        _, v = model.encode(prepare_data(model, x))
        v_list.append(v.detach().cpu().numpy())
    return v_list
