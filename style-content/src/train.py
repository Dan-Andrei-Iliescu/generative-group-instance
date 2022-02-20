import fire
import os
import time
import torch
import numpy as np

from torchvision.io import read_image, write_png, ImageReadMode

from src.model import Model
from utils.helpers import elapsed_time, pack_img, unpack_img


def train(num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "data"
    imgs = []
    for filename in os.listdir(data_dir):
        if filename.endswith((".jpeg", ".png", ".jpg")):
            img = read_image(os.path.join(data_dir, filename),
                             mode=ImageReadMode.RGB)
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

            x_, v_ = model.reconstruct(x)
            x_ = unpack_img(x_.detach())
            result_path = os.path.join(
                "results", "normal", f"img_{epoch}_{idx}.png")
            write_png(x_, result_path)

            """
            v_ = unpack_img(v_.detach())
            result_path = os.path.join(
                "results", "normal", f"latent_{epoch}_{idx}.png")
            write_png(v_, result_path)
            """

            if idx == 0:
                y = x
            xy = model.translate(x, y)
            xy = unpack_img(xy.detach())
            result_path = os.path.join(
                "results", "normal", f"trans_{epoch}_{idx}.png")
            write_png(xy, result_path)
            idx += 1
        epoch_loss /= len(imgs)

        # Current time
        elapsed, mins, secs = elapsed_time(start_time)
        per_epoch = elapsed / (epoch + 1)
        print("> Training epochs [%d/%d] took %dm%ds, %.1fs/epoch" % (epoch, num_epochs, mins, secs, per_epoch)
              + "\nEpoch loss: %.4f" % (epoch_loss))


if __name__ == '__main__':
    fire.Fire(train)
