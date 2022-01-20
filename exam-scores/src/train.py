import fire
import os
import time
import numpy as np
import pandas as pd

from utils.helpers import prepare_data, trans_test, rec_test, latent_test, \
    rec_error, elapsed_time
from utils.plots import plot_trans
from utils.toy_data import generate_dataset
from src.model import Model
from src.latent_pred import LatentPred, LatentPredMulti


def train(
        group_acc=None, inst_cond=True, reg=None,
        num_train_batches=1024, batch_size=64, num_test_batches=128,
        num_epochs=64, test_freq=16, lr=1e-4, result_path="results",
        seed=2, uv_ratio=None):

    # Path to save test results
    result_csv = os.path.join(result_path, "results.csv")
    model_name = f"{inst_cond}_{reg}_{group_acc}_{uv_ratio}"
    result_name = os.path.join(result_path, model_name)

    # Setup datasets
    train_data, test_a, test_b, test_ab = generate_dataset(
        num_train_batches=num_train_batches, num_test_batches=num_test_batches,
        batch_size=batch_size, seed=seed, uv_ratio=uv_ratio)

    train_x = train_data[0]
    train_u = train_data[1]
    train_v = train_data[2]

    test_a_x = test_a[0]
    test_a_u = test_a[1]

    test_b_x = test_b[0]
    test_ab_x = test_ab[0]

    # Dictionary of results to be stored for testing
    test_dict = {}
    test_dict['rec_error'] = {}
    test_dict['trans_error'] = {}
    test_dict['u_error'] = {}
    test_dict['v_error'] = {}

    # Setup the models
    model = Model(
        group_acc=group_acc, inst_cond=inst_cond, reg=reg, lr=lr)
    u_net = LatentPred(lr=lr, z_dim=2)
    v_net = LatentPredMulti(lr=lr, v_dim=1, u_dim=2, h_dim=32)

    # training loop
    dfs = []
    start_time = time.time()
    for epoch in range(num_epochs):
        for x, u, v in zip(train_x, train_u, train_v):
            model.step(prepare_data(x))
            u_inf, v_inf = model.inference(prepare_data(x))
            u_net.step(u_inf.detach(), prepare_data(u))
            v_net.step(v_inf.detach(), prepare_data(u))

        # Testing
        result_disp = result_name + f" @ epoch {epoch+1}"

        # Test reconstruction
        rec_batch = rec_test(model, test_a_x)
        rec_err = rec_error(test_a_x, rec_batch)

        # Test translation
        trans_res = trans_test(model, test_a_x, test_b_x)
        trans_err = rec_error(test_ab_x, trans_res)

        # Plot
        if (epoch+1) % test_freq == 0:
            plot_trans(test_a_x[0], test_b_x[0], trans_res[0], result_name)

        # Test the accuracy of predicting the true latents from the inferred ones
        u_pred, v_pred = latent_test(model, u_net, v_net, test_a_x)
        u_error = rec_error(u_pred, test_a_u)
        v_error = rec_error(v_pred, test_a_u)

        # Current time
        elapsed, mins, secs = elapsed_time(start_time)
        per_epoch = elapsed / (epoch + 1)
        print("> Training [%s / %d] took %dm%ds, %.1fs/epoch" %
              (result_disp, num_epochs, mins, secs, per_epoch) +
              "\nReconstruction error: %.4f" %
              (np.mean(rec_err)) +
              "\nTranslation error: %.4f" %
              (np.mean(trans_err)) +
              "\nGroup prediction error: %.4f, Instance prediction error: %.4f" %
              (np.mean(u_error), np.mean(v_error)))

        # Add these results to the dataframe
        df = pd.DataFrame({
            'test_name': ['rec_error', 'trans_error', 'u_error', 'v_error'],
            'model_name': [model_name, model_name, model_name, model_name],
            'inst_cond': [inst_cond, inst_cond, inst_cond, inst_cond],
            'reg': [reg, reg, reg, reg],
            'group_acc': [group_acc, group_acc, group_acc, group_acc],
            'uv_ratio': [uv_ratio, uv_ratio, uv_ratio, uv_ratio],
            'seed': [seed, seed, seed, seed],
            'epoch': [epoch, epoch, epoch, epoch],
            'value': [rec_err, trans_err, u_error, v_error]})
        dfs.append(df)

    # Save dataframe of results
    test_df = pd.DataFrame(columns=['test_name', 'model_name', 'inst_cond',
                                    'reg', 'group_acc', 'uv_ratio', 'seed',
                                    'epoch', 'value'])
    for df in dfs:
        test_df = pd.concat([test_df, df])
    if os.path.exists(result_csv):
        read_df = pd.read_csv(result_csv, engine='python')
        test_df = pd.concat([read_df, test_df])
    test_df.to_csv(result_csv, index=False)


if __name__ == '__main__':
    fire.Fire(train)
