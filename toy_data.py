import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


def generate_dataset(
        x_dim, num_train_groups, train_lam, num_test_groups, test_lam):

    min_num = 3

    num_instances = min_num + np.random.poisson(lam=train_lam, size=num_train_groups)
    mean_vars = 4 * np.random.normal(
        loc=0, scale=1, size=[num_train_groups, x_dim])
    sdev_vars = 0.5 * np.random.normal(
        loc=0, scale=1, size=[num_train_groups, x_dim])

    train_data = []
    for group_idx in tqdm(range(num_train_groups)):
        inst_vars = np.random.normal(
            loc=0, scale=1, size=[num_instances[group_idx], x_dim])
        x = mean_vars[group_idx] + sdev_vars[group_idx]**2 * inst_vars
        train_data.append(x)

    num_instances = min_num + np.random.poisson(lam=test_lam, size=num_test_groups)
    mean_vars = 4 * np.random.normal(
        loc=0, scale=1, size=[num_test_groups, x_dim])
    sdev_vars = 0.5 * np.random.normal(
        loc=0, scale=1, size=[num_test_groups, x_dim])

    test_data = []
    for group_idx in tqdm(range(num_test_groups)):
        inst_vars = np.random.normal(
            loc=0, scale=1, size=[num_instances[group_idx], x_dim])
        x = mean_vars[group_idx] + sdev_vars[group_idx]**2 * inst_vars
        test_data.append(x)

    return train_data, test_data


if __name__ == '__main__':
    x, y = generate_dataset(
        x_dim=4, num_train_groups=1000, train_lam=8, num_test_groups=100,
        test_lam=16)
    print(x[0:2])
