import torch
import numpy as np
import time


# turn input numpy array into torch tensor
def prepare_data(model, x):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    x = torch.Tensor(x)
    # if on GPU put mini-batch into CUDA memory
    if model.use_cuda:
        x = x.cuda()
    return x


def un_prepare_data(x):
    x = x.detach().cpu().numpy()
    if x.shape[0] == 1:
        x = x[0]
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
        trans.append(un_prepare_data(model.translate(
            prepare_data(model, x), prepare_data(model, y))))
    return trans


def rec_test(model, test_data):
    rec = []
    for x in test_data:
        rec.append(un_prepare_data(model.reconstruct(prepare_data(model, x))))
    return rec


def latent_test(model, test_data):
    v_list = []
    for x in test_data:
        _, v = model.encode(prepare_data(model, x))
        v_list.append(un_prepare_data(v))
    return v_list


def elapsed_time(start_time):
    curr_time = time.time()
    elapsed = curr_time - start_time
    mins = elapsed / 60
    secs = elapsed % 60
    return elapsed, mins, secs
