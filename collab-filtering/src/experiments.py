import os
import fire
import time
import itertools
from joblib import Parallel, delayed
from src.train import train
from utils.results import results
from utils.helpers import elapsed_time


def exp(result_dir="results", exp_name=None, training=True):
    exp_dir = os.path.join(result_dir, exp_name)
    num_epochs = 64

    if training:
        # All possible conditions
        group_acc_vals = [None, "mul", "med"]
        group_acc_def = "mul"

        inst_cond_vals = [True, False]
        inst_cond_def = False

        reg_vals = [None, "nemeth", "ours"]
        reg_def = None

        num_train_batches_vals = [256, 1024, 4096]
        num_train_batches_def = 1024

        batch_size_vals = [16, 64, 256]
        batch_size_def = 64

        lr_vals = [3, 4, 6]
        lr_def = 4

        seed_vals = [2, 8, 32, 128, 512]
        seed_vals = [2, 32, 512]
        # seed_vals = [128]

        uv_ratio_vals = [0.01, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.99]
        uv_ratio_def = 0.5

        xy_ratio_vals = [0., 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 1.]
        xy_ratio_def = 1.

        # Select relevant conditions based on the requested experiment
        cond_dicts = []
        if exp_name == "ablation":
            for seed in seed_vals:
                for group_acc in group_acc_vals:
                    for inst_cond in inst_cond_vals:
                        for reg in reg_vals:
                            dict = {}
                            dict['group_acc'] = group_acc
                            dict['inst_cond'] = inst_cond
                            dict['reg'] = reg
                            dict['num_train_batches'] = num_train_batches_def
                            dict['batch_size'] = batch_size_def
                            dict['lr'] = lr_def
                            dict['seed'] = seed
                            dict['uv_ratio'] = uv_ratio_def
                            dict['xy_ratio'] = xy_ratio_def
                            cond_dicts.append(dict)
        elif exp_name == "hyper_param":
            for seed in seed_vals:
                for num_train_batches in num_train_batches_vals:
                    for batch_size in batch_size_vals:
                        for lr in lr_vals:
                            dict = {}
                            dict['group_acc'] = group_acc_def
                            dict['inst_cond'] = inst_cond_def
                            dict['reg'] = reg_def
                            dict['num_train_batches'] = num_train_batches
                            dict['batch_size'] = batch_size
                            dict['lr'] = lr
                            dict['seed'] = seed
                            dict['uv_ratio'] = uv_ratio_def
                            dict['xy_ratio'] = xy_ratio_def
                            cond_dicts.append(dict)
        elif exp_name == "ours_vs_theirs":
            group_accs = [None, "mul", "med", "med"]
            inst_conds = [True, False, False, False]
            regs = [None, None, None, "nemeth"]

            for seed in seed_vals:
                conds = zip(group_accs, inst_conds, regs)
                for (group_acc, inst_cond, reg) in conds:
                    dict = {}
                    dict['group_acc'] = group_acc
                    dict['inst_cond'] = inst_cond
                    dict['reg'] = reg
                    dict['num_train_batches'] = num_train_batches_def
                    dict['batch_size'] = batch_size_def
                    dict['lr'] = lr_def
                    dict['seed'] = seed
                    dict['uv_ratio'] = uv_ratio_def
                    dict['xy_ratio'] = xy_ratio_def
                    cond_dicts.append(dict)
        elif exp_name == "sub_ablation":
            group_accs = [None, "mul", "med", None, None, None]
            inst_conds = [True, True, True, False, True, True]
            regs = [
                "ours", "ours", "ours", "ours",
                None, "nemeth"]

            for seed in seed_vals:
                conds = zip(group_accs, inst_conds, regs)
                for (group_acc, inst_cond, reg) in conds:
                    dict = {}
                    dict['group_acc'] = group_acc
                    dict['inst_cond'] = inst_cond
                    dict['reg'] = reg
                    dict['num_train_batches'] = num_train_batches_def
                    dict['batch_size'] = batch_size_def
                    dict['lr'] = lr_def
                    dict['seed'] = seed
                    dict['uv_ratio'] = uv_ratio_def
                    dict['xy_ratio'] = xy_ratio_def
                    cond_dicts.append(dict)
        elif exp_name == "just_one":
            for seed in seed_vals:
                dict = {}
                dict['group_acc'] = group_acc_def
                dict['inst_cond'] = inst_cond_def
                dict['reg'] = reg_def
                dict['num_train_batches'] = num_train_batches_def
                dict['batch_size'] = batch_size_def
                dict['lr'] = lr_def
                dict['seed'] = seed
                dict['uv_ratio'] = uv_ratio_def
                dict['xy_ratio'] = xy_ratio_def
                cond_dicts.append(dict)
        elif exp_name == "uv_ratio":
            group_accs = [None, "med"]
            inst_conds = [True, False]
            regs = [None, None]
            for seed in seed_vals:
                for uv_ratio in uv_ratio_vals:
                    conds = zip(group_accs, inst_conds, regs)
                    for (group_acc, inst_cond, reg) in conds:
                        dict = {}
                        dict['group_acc'] = group_acc
                        dict['inst_cond'] = inst_cond
                        dict['reg'] = reg
                        dict['num_train_batches'] = num_train_batches_def
                        dict['batch_size'] = batch_size_def
                        dict['lr'] = lr_def
                        dict['seed'] = seed
                        dict['uv_ratio'] = uv_ratio
                        dict['xy_ratio'] = xy_ratio_def
                        cond_dicts.append(dict)
        elif exp_name == "xy_ratio":
            group_accs = [None, "med"]
            inst_conds = [True, False]
            regs = [None, None]
            for seed in seed_vals:
                for xy_ratio in xy_ratio_vals:
                    conds = zip(group_accs, inst_conds, regs)
                    for (group_acc, inst_cond, reg) in conds:
                        dict = {}
                        dict['group_acc'] = group_acc
                        dict['inst_cond'] = inst_cond
                        dict['reg'] = reg
                        dict['num_train_batches'] = num_train_batches_def
                        dict['batch_size'] = batch_size_def
                        dict['lr'] = lr_def
                        dict['seed'] = seed
                        dict['uv_ratio'] = uv_ratio_def
                        dict['xy_ratio'] = xy_ratio
                        cond_dicts.append(dict)

        # Run in parallel training for all conditions
        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(train)(
            num_epochs=num_epochs,
            group_acc=dict['group_acc'], inst_cond=dict['inst_cond'],
            reg=dict['reg'], num_train_batches=dict['num_train_batches'],
            batch_size=dict['batch_size'], lr=0.1**dict['lr'],
            seed=dict['seed'], uv_ratio=dict['uv_ratio'],
            xy_ratio=dict['xy_ratio'], result_path=exp_dir
        ) for dict in cond_dicts)
        _, mins, secs = elapsed_time(start_time)
        print("\nExperiment %s took %dm%ds to train\n\n" %
              (exp_name, mins, secs))

    # Process results
    results(result_dir=exp_dir, skip=num_epochs-5)


if __name__ == '__main__':
    fire.Fire(exp)
