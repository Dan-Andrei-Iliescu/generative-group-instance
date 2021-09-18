import numpy as np
from tqdm import tqdm
from utils.plots import plot_1D_rec, plot_1D_trans


def generate_dataset(
        x_dim=2, num_train_groups=64, num_test_groups=8, min_num=8, lam=16):
    mean = 2
    scale = 1

    # Training data
    print("TRAINING DATA")

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

    # Testing x
    print("TESTING X")

    num_instances = min_num + np.random.poisson(lam=lam, size=num_test_groups)
    mean_vars = np.random.normal(
        loc=0, scale=mean, size=[num_train_groups, x_dim])
    sdev_vars = np.abs(np.random.normal(
        loc=0, scale=scale, size=[num_train_groups, x_dim]))**2

    test_x = []
    inst_vars = []
    for group_idx in tqdm(range(num_test_groups)):
        inst_vars.append(np.random.normal(
            loc=0, scale=1, size=[num_instances[group_idx], x_dim]))
        x = mean_vars[group_idx] + sdev_vars[group_idx] * inst_vars[group_idx]
        test_x.append(x)

    # Testing translation (shares inst vars with x)
    print("TESTING TRANSLATION")

    mean_vars = np.random.normal(
        loc=0, scale=mean, size=[num_train_groups, x_dim])
    sdev_vars = np.abs(np.random.normal(
        loc=0, scale=scale, size=[num_train_groups, x_dim]))**2

    test_trans = []
    for group_idx in tqdm(range(num_test_groups)):
        x = mean_vars[group_idx] + sdev_vars[group_idx] * inst_vars[group_idx]
        test_trans.append(x)

    # Testing y (shares mean and sdev with trans)
    print("TESTING Y")

    num_instances = min_num + np.random.poisson(lam=lam, size=num_test_groups)

    test_y = []
    for group_idx in tqdm(range(num_test_groups)):
        inst_vars = np.random.normal(
            loc=0, scale=1, size=[num_instances[group_idx], x_dim])

        x = mean_vars[group_idx] + sdev_vars[group_idx] * inst_vars
        test_y.append(x)

    return train_data, test_x, test_y, test_trans


if __name__ == '__main__':
    x, y = generate_dataset()

    plot_1D_rec(y, y, "Test_rec", "test_rec")
    plot_1D_trans(x[0], x[1], x[2], "Test_trans", "test_trans")
