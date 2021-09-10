import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from utils.plots import plot_1D_rec, plot_1D_trans


def generate_dataset(
        x_dim=2, num_train_groups=64, num_test_groups=8, min_num=8, lam=16):
    mean = 2
    scale = 1

    np.random.seed(100)
    num_instances = min_num + np.random.poisson(lam=lam, size=num_train_groups)
    mean_vars = np.random.normal(
        loc=0, scale=mean, size=[num_train_groups, x_dim])
    sdev_vars = np.abs(np.random.normal(
        loc=0, scale=scale, size=[num_train_groups, x_dim]))**2

    train_data = []
    for group_idx in tqdm(range(num_train_groups)):
        inst_vars = np.random.normal(
            loc=0, scale=1, size=[num_instances[group_idx], x_dim])

        x = mean_vars[group_idx] + sdev_vars[group_idx] * inst_vars
        train_data.append(x)

    num_instances = min_num + np.random.poisson(lam=lam, size=num_test_groups)
    mean_vars = np.random.normal(
        loc=0, scale=mean, size=[num_train_groups, x_dim])
    sdev_vars = np.abs(np.random.normal(
        loc=0, scale=scale, size=[num_train_groups, x_dim]))**2

    test_data = []
    for group_idx in tqdm(range(num_test_groups)):
        inst_vars = np.random.normal(
            loc=0, scale=1, size=[num_instances[group_idx], x_dim])

        x = mean_vars[group_idx] + sdev_vars[group_idx] * inst_vars
        test_data.append(x)

    return train_data, test_data


if __name__ == '__main__':
    x, y = generate_dataset()

    plot_1D_rec(y, y, "Test_rec", "test_rec")
    plot_1D_trans(x[0], x[1], x[2], "Test_trans", "test_trans")
