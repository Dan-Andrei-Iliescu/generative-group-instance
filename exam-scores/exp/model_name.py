import os
import fire
import time
from joblib import Parallel, delayed
from src.train import train
from utils.results import results
from utils.helpers import elapsed_time


def exp(result_dir="results", training=True):
    exp_dir = os.path.join(result_dir, "model_name")

    if training:
        model_names = ["ml_vae", "ml_vae_bad_cond"]
        num_training_batches = 512
        batch_size = 256

        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(train)(
            model_name=model_name,
            num_train_batches=num_training_batches, batch_size=batch_size,
            result_path=os.path.join(exp_dir, model_name)
        ) for model_name in model_names)
        _, mins, secs = elapsed_time(start_time)
        print("\nExperiment model name took %dm%ds to train" % (mins, secs))

    results(result_dir=exp_dir)


if __name__ == '__main__':
    fire.Fire(exp)
