import fire
import json
import os
import time
import numpy as np

from utils.helpers import prepare_data, trans_test, rec_test, latent_test, \
    rec_error, elapsed_time
from utils.plots import plot_1D_trans
from utils.toy_data import generate_dataset
from src.model import Model
from src.latent_pred import LatentPred


def train(
        group_acc=None, inst_cond=True, reg=None,
        num_train_batches=1024, batch_size=64, num_test_batches=128,
        num_epochs=64, test_freq=16, lr=1e-4, result_path=None, seed=2):

    # Path to save test results
    if result_path is None:
        result_path = os.path.join("results", f"{group_acc}_{inst_cond}_{reg}")
    result_prog_path = result_path + "_prog"

    # Setup datasets
    train_data, test_a, test_b, test_ab = generate_dataset(
        num_train_batches=num_train_batches, num_test_batches=num_test_batches,
        batch_size=batch_size, seed=seed)

    train_x = train_data[0]
    train_u = train_data[1]
    train_v = train_data[2]

    test_a_x = test_a[0]
    test_a_u = test_a[1]
    test_a_v = test_a[2]

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
    v_net = LatentPred(lr=lr, z_dim=1)

    # training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        for x, u, v in zip(train_x, train_u, train_v):
            model.step(prepare_data(x))
            u_inf, v_inf = model.inference(prepare_data(x))
            u_net.step(u_inf.detach(), prepare_data(u))
            v_net.step(v_inf.detach(), prepare_data(v))

        # Testing
        result_name = result_path + f" @ epoch {epoch+1}"

        # Test reconstruction
        rec_batch = rec_test(model, test_a_x)
        rec_err = rec_error(test_a_x, rec_batch)
        test_dict['rec_error'][epoch] = rec_err

        # Test translation
        trans_res = trans_test(model, test_a_x, test_b_x)
        trans_err = rec_error(test_ab_x, trans_res)
        test_dict['trans_error'][epoch] = trans_err

        # Plot
        if (epoch+1) % test_freq == 0:
            plot_1D_trans(test_a_x[0], test_b_x[0], trans_res[0],
                          result_name, result_path)

        # Test the accuracy of predicting the true latents from the inferred ones
        u_pred, v_pred = latent_test(model, u_net, v_net, test_a_x)
        u_error = rec_error(u_pred, test_a_u)
        v_error = rec_error(v_pred, test_a_v)
        # Ridiculous but necessary for the purpose of serialization
        u_error = [float(x) for x in u_error]
        v_error = [float(x) for x in v_error]
        test_dict['u_error'][epoch] = u_error
        test_dict['v_error'][epoch] = v_error

        # Current time
        elapsed, mins, secs = elapsed_time(start_time)
        per_epoch = elapsed / (epoch + 1)
        print("> Training [%s / %d] took %dm%ds, %.1fs/epoch" %
              (result_name, num_epochs, mins, secs, per_epoch) +
              "\nReconstruction error: %.4f" %
              (np.mean(rec_err)) +
              "\nTranslation error: %.4f" %
              (np.mean(trans_err)) +
              "\nGroup prediction error: %.4f, Instance prediction error: %.4f" %
              (np.mean(u_error), np.mean(v_error)))

        # Save json file of test results
        jsonString = json.dumps(test_dict)
        with open(result_prog_path, "w") as jsonFile:
            jsonFile.write(jsonString)

    # Add this run to the other runs
    if os.path.exists(result_path):
        # Read test dict
        fileObject = open(result_path, "r")
        jsonContent = fileObject.read()
        runs_dict = json.loads(jsonContent)
    else:
        runs_dict = {}
    plot_names = list(test_dict.keys())
    for plot_name in plot_names:
        if plot_name not in runs_dict:
            runs_dict[plot_name] = {}
        for epoch in range(num_epochs):
            if str(epoch) in runs_dict[plot_name]:
                runs_dict[plot_name][str(epoch)].append(
                    test_dict[plot_name][epoch])
            else:
                runs_dict[plot_name][epoch] = [test_dict[plot_name][epoch]]

    # Save json file of test results
    jsonString = json.dumps(runs_dict)
    with open(result_path, "w") as jsonFile:
        jsonFile.write(jsonString)

    # Remove progress file
    if os.path.exists(result_prog_path):
        os.remove(result_prog_path)


if __name__ == '__main__':
    fire.Fire(train)
