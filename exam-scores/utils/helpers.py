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
    return error


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


def moving_avg(a, n):
    s = np.cumsum(a)
    s[n:] = s[n:] - s[:-n]
    a[n-1:] = s[n-1:] / n
    return a


def save_results(test_dict, result_dir):
    df_dict = {}
    cond_names = list(test_dict.keys())
    for cond_name in cond_names:
        conds = cond_name.split("-")
        for idx in range(len(conds)):
            if f"cond_{idx}" not in df_dict:
                df_dict[f"cond_{idx}"] = [conds[idx]]
            else:
                df_dict[f"cond_{idx}"].append(conds[idx])

        test_names = list(test_dict[cond_name].keys())
        test_vals = []
        for test_name in test_names:
            epochs = list(test_dict[cond_name][test_name].keys())
            error_mean = []
            error_sdev = []
            for epoch in epochs:
                runs = test_dict[cond_name][test_name][epoch]
                run_means = [np.mean(run) for run in runs]
                error_mean.append(np.mean(run_means))
                error_sdev.append(np.sqrt(np.mean(
                    (np.mean(run_means) - run_means)**2)))

            n = 20
            dec = 1e+2
            error_mean = dec * np.mean(np.array(error_mean)[-n:])
            error_sdev = dec * np.mean(np.array(error_sdev)[-n:])

            test_name_mean = test_name+"_mean"
            test_name_sdev = test_name+"_sdev"
            test_vals.append(test_name_mean)
            test_vals.append(test_name_sdev)

            if test_name_mean in df_dict:
                df_dict[test_name_mean].append(error_mean)
            else:
                df_dict[test_name_mean] = [error_mean]

            if test_name_sdev in df_dict:
                df_dict[test_name_sdev].append(error_sdev)
            else:
                df_dict[test_name_sdev] = [error_sdev]

    df = pd.DataFrame(data=df_dict)
    df.apply(my_round).to_csv(
        os.path.join(result_dir, "table_results.csv"))

    for idx in range(len(conds)):
        agg_df = df.groupby([f"cond_{idx}"])[test_vals].mean()
        agg_df.apply(my_round).to_csv(
            os.path.join(result_dir, f"cond_{idx}.csv"))
