import pyro
import fire
import json
import os
import time
import numpy as np

from utils.helpers import prepare_data, trans_test, rec_test, latent_test, \
    rec_error, latent_error, elapsed_time
from utils.plots import plot_1D_trans, plot_1D_latent
from utils.toy_data import generate_dataset
from src.model import Model


def train(
        group_acc=None, inst_cond=True, reg=None,  cuda=False, wd=1e-6,
        num_train_batches=256, batch_size=64, num_test_groups=128,
        num_epochs=40, test_freq=20, lr=1e-3, result_path=None):
    x_dim = 1

    # Path to save test results
    if result_path is None:
        result_path = os.path.join("results", f"{group_acc}_{inst_cond}_{reg}")
    result_prog_path = result_path + "_prog"

    # clear param store
    pyro.clear_param_store()

    # setup data lists
    train_data, test_x, test_y, test_trans = generate_dataset(
        x_dim=x_dim, num_train_batches=num_train_batches,
        batch_size=batch_size, num_test_groups=num_test_groups)

    # dictionary of results to be stored for testing
    test_dict = {}
    test_dict['rec_error'] = {}
    test_dict['trans_error'] = {}
    test_dict['latent_mean'] = {}
    test_dict['latent_var'] = {}

    # setup the model
    model = Model(
        group_acc=group_acc, inst_cond=inst_cond, reg=reg, x_dim=x_dim,
        lr=lr, cuda=cuda, wd=wd)

    # training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x in train_data:
            # do ELBO gradient and accumulate loss
            epoch_loss += model.step(prepare_data(model, x))

        """
        # report training diagnostics
        normalizer_train = len(train_data)
        total_epoch_loss_train = epoch_loss / normalizer_train
        print("> [epoch %03d]  Training loss: %.4f" %
              (epoch, total_epoch_loss_train))
        """

        # Testing
        result_name = result_path + f" @ epoch {epoch+1}"

        # current time
        elapsed, mins, secs = elapsed_time(start_time)
        per_epoch = elapsed / (epoch + 1)
        print("> Training [%s / %d] took %dm%ds, %.1fs/epoch" %
              (result_name, num_epochs, mins, secs, per_epoch))

        # test reconstruction
        rec_batch = rec_test(model, test_x)
        rec_err = rec_error(test_x, rec_batch)
        print("[epoch %03d]  reconstruction error: %.4f" %
              (epoch, np.sum(rec_err)))
        test_dict['rec_error'][epoch] = rec_err

        # test translation
        trans_batch = trans_test(model, test_x, test_y)
        trans_err = rec_error(test_trans, trans_batch)
        print("[epoch %03d]  translation error: %.4f" %
              (epoch, np.sum(trans_err)))
        test_dict['trans_error'][epoch] = trans_err

        # Plot
        if (epoch+1) % test_freq == 0:
            plot_1D_trans(test_x, test_y, trans_batch, result_name)

        # test latents
        v_batch = latent_test(model, test_x)
        mean_err, var_err = latent_error(v_batch)
        # Ridiculous but necessary for the purpose of serialization
        mean_err = [float(x) for x in mean_err]
        var_err = [float(x) for x in var_err]
        print("[epoch %03d]  latent mean: %.4f, latent var: %.4f" %
              (epoch, np.sum(mean_err), np.sum(var_err)))
        test_dict['latent_mean'][epoch] = mean_err
        test_dict['latent_var'][epoch] = var_err

        # Plot
        if (epoch+1) % test_freq == 0:
            plot_1D_latent(v_batch, result_name)

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
