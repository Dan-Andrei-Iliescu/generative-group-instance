import os
import fire
import time
import itertools
from joblib import Parallel, delayed
from src.train import train
from utils.results import results
from utils.helpers import elapsed_time


def exp(result_dir="results", exp_name="final", training=True, runs=8):
    exp_dir = os.path.join(result_dir, exp_name)

    if training:
        # All possible conditions
        group_acc = ["mul", "med"]
        group_acc_def = ["mul"]

        inst_cond = [True, False]
        inst_cond_def = [False]
        inst_cond_def = [True]

        reg = [None, "nemeth", "nemeth_group"]
        reg_def = [None]

        num_train_batches = [1024, 2048, 4096]
        num_train_batches_def = [1024]

        batch_size = [64, 128, 256]
        batch_size_def = [64]

        lr = [3, 4, 6]
        lr_def = [4]

        seeds = [2, 8, 32, 128, 512]

        # Select relevant conditions based on the requested experiment
        serial_conds = [
            group_acc_def, inst_cond_def, reg_def, num_train_batches_def,
            batch_size_def, lr_def, seeds]
        if exp_name == "model_name":
            serial_conds = [
                group_acc, inst_cond, reg, num_train_batches_def,
                batch_size_def, lr_def, seeds]
        elif exp_name == "hyper_param":
            serial_conds = [
                group_acc_def, inst_cond_def, reg_def, num_train_batches,
                batch_size, lr, seeds]
        elif exp_name == "batch_size":
            serial_conds = [
                group_acc_def, inst_cond_def, reg_def, num_train_batches_def,
                batch_size, lr_def, seeds]
        elif exp_name == "num_train_batches":
            serial_conds = [
                group_acc_def, inst_cond_def, reg_def, num_train_batches,
                batch_size_def, lr_def, seeds]
        elif exp_name == "lr":
            serial_conds = [
                group_acc_def, inst_cond_def, reg_def, num_train_batches_def,
                batch_size_def, lr, seeds]
        elif exp_name == "cond":
            serial_conds = [
                group_acc_def, inst_cond, reg_def, num_train_batches_def,
                batch_size_def, lr_def, seeds]
        elif exp_name == "final":
            serial_conds = [
                group_acc_def, inst_cond, reg, num_train_batches_def,
                batch_size_def, lr_def, seeds]

        # Match every selected condition with every other condition
        cross_conds = []
        for _ in range(runs):
            cross_conds += list(itertools.product(*serial_conds))

        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(train)(
            group_acc=cond[0], inst_cond=cond[1], reg=cond[2],
            num_train_batches=cond[3], batch_size=cond[4], lr=0.1**cond[5],
            seed=cond[6], result_path=os.path.join(
                exp_dir, "%s-%s-%s-%04d-%03d-%s" %
                (cond[0], cond[1], cond[2], cond[3], cond[4], cond[5]))
        ) for cond in cross_conds)
        _, mins, secs = elapsed_time(start_time)
        print("\nExperiment %s took %dm%ds to train\n\n" %
              (exp_name, mins, secs))
    results(result_dir=exp_dir)


if __name__ == '__main__':
    fire.Fire(exp)
