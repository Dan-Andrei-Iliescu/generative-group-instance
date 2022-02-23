import fire
import os
import time
import torch

import numpy as np

from src.model import Model
from utils.helpers import pack_img, unpack_img, read_img, write_img
from utils.helpers import elapsed_time


def train(num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "data"
    imgs = []
    for filename in os.listdir(data_dir):
        if filename.endswith((".jpeg", ".png", ".jpg")):
            img = read_img(os.path.join(data_dir, filename))
            img = pack_img(img, device)
            imgs.append(img)

    # setup the model
    model = Model().to(device)

    # training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.
        idx = 0
        for x in imgs:
            # do ELBO gradient and accumulate loss
            epoch_loss += model.step(x)
        epoch_loss /= len(imgs)

        # Random images to plot
        x_idx = np.random.randint(0, len(imgs))
        y_idx = np.random.randint(0, len(imgs))
        x = imgs[x_idx]
        y = imgs[y_idx]

        x_, v_ = model.reconstruct(x)
        x_ = unpack_img(x_.detach())
        result_path = os.path.join(
            "results", "normal", f"img_{epoch}_{x_idx}.png")
        write_img(x_, result_path)

        m = torch.randn(3, v_.shape[-1])
        v_ = torch.einsum('xv,nijv->nijx', m, v_)
        v_ = unpack_img(v_.detach())
        result_path = os.path.join(
            "results", "normal", f"latent_{epoch}_{x_idx}.png")
        write_img(v_, result_path)

        xy = model.translate(x, y)
        xy = unpack_img(xy.detach())
        result_path = os.path.join(
            "results", "normal", f"trans_{epoch}_{x_idx}_{y_idx}.png")
        write_img(xy, result_path)
        idx += 1

        # Current time
        elapsed, mins, secs = elapsed_time(start_time)
        per_epoch = elapsed / (epoch + 1)
        print("> Training epochs [%d/%d] took %dm%ds, %.1fs/epoch" % (epoch, num_epochs, mins, secs, per_epoch)
              + "\nEpoch loss: %.4f" % (epoch_loss))


if __name__ == '__main__':
    fire.Fire(train)
