import torch
import time
import os
import numpy as np
import pandas as pd


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
        _, v = model.inference(prepare_data(model, x))
        v_list.append(un_prepare_data(v))
    return v_list


def elapsed_time(start_time):
    curr_time = time.time()
    elapsed = curr_time - start_time
    mins = elapsed / 60
    secs = elapsed % 60
    return elapsed, mins, secs


def my_round(x):
    try:
        return np.around(x, decimals=3)
    except:
        return x


def compute_results(test_dict, result_dir):
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
            coeff = 0.5
            error_mean = np.mean(np.array(error_mean)[-n:])
            error_sdev = coeff * np.mean(np.array(error_sdev)[-n:])

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
