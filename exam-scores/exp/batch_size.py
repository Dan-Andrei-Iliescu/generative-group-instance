import os
import fire
import time
from joblib import Parallel, delayed
from src.train import train
from utils.results import results


def batch_size_exp(result_dir="results", training=True):
    exp_dir = os.path.join(result_dir, "batch_size")

    if training:
        model_name = "ml_vae"
        conds = [[64, 512],
                 [256, 128]]

        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(train)(
            num_train_batches=cond[0], batch_size=cond[1],
            result_path=os.path.join(
                exp_dir, model_name+"_%04d_%04d" % (cond[0], cond[1]))
        ) for cond in conds)
        elapsed = time.time() - start_time
        print("\nExperiment batch size took %.1fs to train" % (elapsed))

    results(result_dir=exp_dir)


if __name__ == '__main__':
    fire.Fire(batch_size_exp)
