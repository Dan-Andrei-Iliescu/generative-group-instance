import torch
import time
import os
import numpy as np
import pandas as pd


def trig(num_px, num_bits):
    rows = torch.zeros([num_px, 2*num_bits])
    r = torch.arange(num_px).unsqueeze(-1)
    r = r / num_px**(torch.arange(num_bits) / num_bits)
    rows[:, 0::2] = torch.sin(r)
    rows[:, 1::2] = torch.cos(r)
    return rows


def trig_pos_emb(x, num_bits):
    rows = trig(x.shape[1], num_bits)
    rows = rows.unsqueeze(1).unsqueeze(0)
    rows = rows.expand(x.shape[0], -1, x.shape[2], -1)

    cols = trig(x.shape[2], num_bits)
    cols = cols.unsqueeze(0).unsqueeze(0)
    cols = cols.expand(x.shape[0], x.shape[1], -1, -1)

    if x.is_cuda:
        rows = rows.cuda(x.get_device())
        cols = cols.cuda(x.get_device())
    return torch.cat([x, rows, cols], axis=-1)


def int_to_binary(x, bits):
    y = torch.ones_like(x)
    mask = 2**torch.arange(bits-1)
    x = x.unsqueeze(-1)
    y = torch.ones_like(x)
    x = x.bitwise_and(mask).byte()
    x = torch.cat([x, y], dim=-1)
    return x


def bin_pos_emb(x, num_bits):
    rows = int_to_binary(torch.arange(x.shape[1]), num_bits)
    rows = rows.unsqueeze(1).unsqueeze(0)
    rows = rows.expand(x.shape[0], -1, x.shape[2], -1)

    cols = int_to_binary(torch.arange(x.shape[2]), num_bits)
    cols = cols.unsqueeze(0).unsqueeze(0)
    cols = cols.expand(x.shape[0], x.shape[1], -1, -1)

    if x.is_cuda:
        rows = rows.cuda(x.get_device())
        cols = cols.cuda(x.get_device())
    return torch.cat([x, rows, cols], axis=-1)


def pack_img(img, device):
    img = img.to(device)
    img = img / 127.5 - 1.
    img = img.movedim(0, 2).unsqueeze(0)
    return img


def unpack_img(x):
    x = x.movedim(3, 1)[0]
    x = x - torch.min(x)
    x = x / torch.max(x)
    x = x * 255.
    x = x.to('cpu', dtype=torch.uint8)
    return x


def elapsed_time(start_time):
    curr_time = time.time()
    elapsed = curr_time - start_time
    mins = elapsed / 60
    secs = elapsed % 60
    return elapsed, mins, secs
