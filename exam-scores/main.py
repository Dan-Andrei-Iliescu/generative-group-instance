import pyro
import fire
import json
import os
import numpy as np
from tqdm import tqdm

from utils.helpers import prepare_data, trans_test, rec_test, latent_test, \
    rec_error, latent_error
from utils.toy_data import generate_dataset
from src.ml_vae import Model as ml_vae
from src.v_vs_n import Model as v_vs_n
from src.v_vs_n_gp import Model as v_vs_n_gp


def main(
        model_name="ml_vae", x_dim=1, num_epochs=40, test_freq=1, lr=1e-4,
        cuda=False, num_train_groups=100000, num_test_groups=32,
        result_dir="results/num_groups"):
    # Defensive
    assert(x_dim == 1)

    # clear param store
    pyro.clear_param_store()

    # setup data lists
    train_data, test_x, test_y, test_trans = generate_dataset(
        x_dim=x_dim, num_train_groups=num_train_groups,
        num_test_groups=num_test_groups)

    # dictionary of results to be stored for testing
    test_dict = {}
    test_dict['rec_error'] = {}
    test_dict['trans_error'] = {}
    test_dict['latent_mean'] = {}
    test_dict['latent_var'] = {}
    result_path = os.path.join(result_dir, model_name)
    result_prog_path = os.path.join(result_dir, model_name + "_prog")

    # setup the model
    if model_name == "ml_vae":
        model = ml_vae(x_dim=x_dim, cuda=cuda, lr=lr)
    elif model_name == "v_vs_n":
        model = v_vs_n(x_dim=x_dim, cuda=cuda, lr=lr)
    elif model_name == "v_vs_n_gp":
        model = v_vs_n_gp(x_dim=x_dim, cuda=cuda, lr=lr)
    else:
        raise NotImplementedError(f"Model type {model_name} does not exist")

    # training loop
    for epoch in tqdm(range(num_epochs)):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x in train_data:
            # do ELBO gradient and accumulate loss
            epoch_loss += model.step(prepare_data(model, x))

        # report training diagnostics
        normalizer_train = len(train_data)
        total_epoch_loss_train = epoch_loss / normalizer_train
        print("> [epoch %03d]  Training loss: %.4f" %
              (epoch, total_epoch_loss_train))

        # Testing
        if epoch % test_freq == 0:
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

            # Save json file of test results
            jsonString = json.dumps(test_dict)
            with open(result_prog_path, "w") as jsonFile:
                jsonFile.write(jsonString)

    # Save json file of test results
    jsonString = json.dumps(test_dict)
    with open(result_path, "w") as jsonFile:
        jsonFile.write(jsonString)


if __name__ == '__main__':
    fire.Fire(main)
