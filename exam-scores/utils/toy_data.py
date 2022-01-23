import fire
import time
import numpy as np
from utils.plots import plot_data, plot_trans
from utils.helpers import elapsed_time


def gt_factors(num_batches, batch_size, rng):
    lam = 16
    min_num = 8

    num_instances = min_num + \
        rng.poisson(lam=lam, size=num_batches)
    u = rng.normal(
        loc=0, scale=1, size=[num_batches, batch_size, 1, 2])
    v = []
    for batch_idx in range(num_batches):
        v.append(rng.normal(
            loc=0, scale=1,
            size=[batch_size, num_instances[batch_idx], 2]))
    return u, v


def generator(u, v, uv_ratio, xy_ratio):
    num_batches = len(v)
    x = []
    for batch_idx in range(num_batches):
        val = np.zeros_like(v[batch_idx])
        val[:, :, 0] = uv_ratio * u[batch_idx, :, :, 0] \
            + (u[batch_idx, :, :, 1]**2 + 1)**uv_ratio \
            * (1 - uv_ratio) * xy_ratio * v[batch_idx][:, :, 0]
        val[:, :, 1] = (1 - xy_ratio) * v[batch_idx][:, :, 1]
        x.append(val)
    return x


def generate_dataset(
        num_train_batches=64, num_test_batches=8, batch_size=32, seed=2,
        uv_ratio=0.5, xy_ratio=1.):

    # Random number generator for this run
    rng = np.random.default_rng(seed)

    # Training data
    start_time = time.time()

    train_u, train_v = gt_factors(num_train_batches, batch_size, rng)
    train_x = generator(train_u, train_v, uv_ratio, xy_ratio)
    train_data = [train_x, train_u, train_v]

    _, mins, secs = elapsed_time(start_time)
    print("Training dataset took %dm%ds to generate" %
          (mins, secs))

    # Testing A
    start_time = time.time()

    test_a_u, test_a_v = gt_factors(num_test_batches, batch_size, rng)
    train_a_x = generator(test_a_u, test_a_v, uv_ratio, xy_ratio)
    test_a = [train_a_x, test_a_u, test_a_v]

    _, mins, secs = elapsed_time(start_time)
    print("Testing dataset A took %dm%ds to generate" %
          (mins, secs))

    # Testing translation (shares inst vars with x)
    start_time = time.time()

    test_ab_u, _ = gt_factors(num_test_batches, batch_size, rng)
    train_ab_x = generator(test_ab_u, test_a_v, uv_ratio, xy_ratio)
    test_ab = [train_ab_x, test_ab_u, test_a_v]

    _, mins, secs = elapsed_time(start_time)
    print("Testing dataset translation took %dm%ds to generate" %
          (mins, secs))

    # Testing y (shares mean and sdev with trans)
    start_time = time.time()

    _, test_b_v = gt_factors(num_test_batches, batch_size, rng)
    train_b_x = generator(test_ab_u, test_b_v, uv_ratio, xy_ratio)
    test_b = [train_b_x, test_ab_u, test_b_v]

    _, mins, secs = elapsed_time(start_time)
    print("Testing dataset B took %dm%ds to generate" %
          (mins, secs))

    return train_data, test_a, test_b, test_ab


if __name__ == '__main__':
    train_data, test_a, test_b, test_ab = generate_dataset(uv_ratio=0.99)
    plot_data(test_a[0][0], "results/data")
    plot_trans(test_a[0][0], test_b[0][0], test_ab[0][0], "results/data")
