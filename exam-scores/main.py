import pyro
import fire

from utils.helpers import prepare_data, trans_test, rec_test, latent_test
from utils.toy_data import generate_dataset
from src.ml_vae import Model as ml_vae
from src.v_vs_n import Model as v_vs_n


def main(
        model_name="ml_vae", x_dim=1, num_epochs=9, test_freq=2, lr=1e-3,
        cuda=False, num_train_groups=8000, num_test_groups=6):
    # clear param store
    pyro.clear_param_store()

    # setup data lists
    train_data, test_x, test_y, test_trans = generate_dataset(
        x_dim=x_dim, num_train_groups=num_train_groups,
        num_test_groups=num_test_groups)

    # setup the model
    if model_name == "ml_vae":
        model = ml_vae(x_dim=x_dim, cuda=cuda, lr=lr)
    elif model_name == "v_vs_n":
        model = v_vs_n(x_dim=x_dim, cuda=cuda, lr=lr)

    # training loop
    for epoch in range(num_epochs):
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
            # plot reconstruction
            rec_test(model, test_x, epoch)
            # plot translation
            trans_test(model, test_x, test_y, test_trans, epoch)
            # plot latents
            latent_test(model, test_x, epoch)


if __name__ == '__main__':
    fire.Fire(main)
