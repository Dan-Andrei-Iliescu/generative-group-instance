from warnings import resetwarnings
import torch
import time
import os
import numpy as np
import pandas as pd


def prepare_data(x):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    x = torch.Tensor(x)
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
    return np.mean(error)


def trans_test(model, test_x, test_y):
    trans = []
    for x, y in zip(test_x, test_y):
        trans.append(un_prepare_data(model.translate(
            prepare_data(x), prepare_data(y))))
    return trans


def rec_test(model, test_data):
    rec = []
    for x in test_data:
        rec.append(un_prepare_data(model.reconstruct(prepare_data(x))))
    return rec


def latent_test(model, u_net, v_net, test_x):
    u_pred = []
    v_pred = []
    for x in test_x:
        u_inf, v_inf = model.inference(prepare_data(x))
        u_pred.append(un_prepare_data(u_net(u_inf)))
        v_pred.append(un_prepare_data(v_net(v_inf)))
    return u_pred, v_pred


def elapsed_time(start_time):
    curr_time = time.time()
    elapsed = curr_time - start_time
    mins = elapsed / 60
    secs = elapsed % 60
    return elapsed, mins, secs


def my_round(x):
    try:
        return np.around(x, decimals=1)
    except:
        return x


def save_results(df, result_dir):
    df = df.loc[df['epoch'] >= 40]
    df = df.groupby(by=['test_name', 'inst_cond', 'reg', 'group_acc'],
                    dropna=False)['value'].mean()
    df.apply(my_round).to_csv(
        os.path.join(result_dir, "final_results.csv"), index=False)
