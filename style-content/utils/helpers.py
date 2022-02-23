import torch
import time
import cv2


def trig(num_px, num_bits):
    rows = torch.zeros([num_px, 2*num_bits])
    r = torch.arange(num_px).unsqueeze(-1)
    r = r / num_px**(torch.arange(num_bits) / num_bits)
    rows[:, 0::2] = torch.sin(r)
    rows[:, 1::2] = torch.cos(r)
    return rows


def trig_pos_emb(x, r_dim):
    r_dim = int(r_dim / 4)

    rows = trig(x.shape[1], r_dim)
    rows = rows.unsqueeze(1).unsqueeze(0)
    rows = rows.expand(x.shape[0], -1, x.shape[2], -1)

    cols = trig(x.shape[2], r_dim)
    cols = cols.unsqueeze(0).unsqueeze(0)
    cols = cols.expand(x.shape[0], x.shape[1], -1, -1)

    if x.is_cuda:
        rows = rows.cuda(x.get_device())
        cols = cols.cuda(x.get_device())
    return torch.cat([rows, cols], axis=-1)


def int_to_binary(x, bits):
    y = torch.ones_like(x)
    mask = 2**torch.arange(bits-1)
    x = x.unsqueeze(-1)
    y = torch.ones_like(x)
    x = x.bitwise_and(mask).byte()
    x = torch.cat([x, y], dim=-1)
    return x.to(torch.float)


def bin_pos_emb(x, r_dim):
    r_dim = int(r_dim / 2)

    rows = int_to_binary(torch.arange(x.shape[1]), r_dim)
    rows = rows.unsqueeze(1).unsqueeze(0)
    rows = rows.expand(x.shape[0], -1, x.shape[2], -1)

    cols = int_to_binary(torch.arange(x.shape[2]), r_dim)
    cols = cols.unsqueeze(0).unsqueeze(0)
    cols = cols.expand(x.shape[0], x.shape[1], -1, -1)

    if x.is_cuda:
        rows = rows.cuda(x.get_device())
        cols = cols.cuda(x.get_device())
    return torch.cat([rows, cols], axis=-1)


def pack_img(img, device):
    img = torch.Tensor(img).to(device)
    img = img / 127.5 - 1.
    img = img.unsqueeze(0)
    return img


def unpack_img(x):
    x = x[0]
    x = x - torch.min(x)
    x = x / torch.max(x)
    x = x * 255.
    x = x.to('cpu', dtype=torch.uint8).numpy()
    return x


def elapsed_time(start_time):
    curr_time = time.time()
    elapsed = curr_time - start_time
    mins = elapsed / 60
    secs = elapsed % 60
    return elapsed, mins, secs


def read_img(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return lab_img


def write_img(img, filename):
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    cv2.imwrite(filename, img)
