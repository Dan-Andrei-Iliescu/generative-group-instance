import glob
import fire
import json
import os
import pandas as pd

from utils.plots import plot_results as plot
from utils.helpers import save_results


def results(result_dir="results", palette=None):
    result_path = os.path.join(result_dir, "results.csv")
    test_df = pd.read_csv(result_path)

    plot(test_df, result_dir, palette)
    # save_results(test_df, result_dir)


if __name__ == '__main__':
    fire.Fire(results)
