import fire
import os
import time
import torch
import numpy as np

from torchvision.io import read_image, write_png, ImageReadMode

from src.model import Model
from utils.helpers import elapsed_time


def train(num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "data"
    imgs = []
    for filename in os.listdir(data_dir):
        if filename.endswith((".jpeg", ".png", ".jpg")):
            img = read_image(os.path.join(data_dir, filename),
                             mode=ImageReadMode.RGB)
            img = img.to(device)
            img = img / 127.5 - 1.
            img = img.movedim(0, 2).unsqueeze(0)
            imgs.append(img)

    # setup the model
    model = Model().to(device)

    # training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.
        for x in imgs:
            # do ELBO gradient and accumulate loss
            epoch_loss += model.step(x)
        epoch_loss /= len(imgs)

        # Current time
        elapsed, mins, secs = elapsed_time(start_time)
        per_epoch = elapsed / (epoch + 1)
        print("> Training epochs [%d/%d] took %dm%ds, %.1fs/epoch" % (epoch, num_epochs, mins, secs, per_epoch)
              + "\nEpoch loss: %.4f" % (epoch_loss))

        x = imgs[epoch % len(imgs)]
        x_ = model.reconstruct(x)[0].detach().movedim(2, 0)
        x_ = x_ - torch.min(x_)
        x_ = x_ / torch.max(x_)
        x_ = x_ * 255.
        x_ = x_.to('cpu', dtype=torch.uint8)
        result_path = os.path.join("results", "normal", f"img_{epoch}.png")
        write_png(x_, result_path)


if __name__ == '__main__':
    fire.Fire(train)
