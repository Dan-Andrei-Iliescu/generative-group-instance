import glob
import fire
import json
import os
import pandas as pd
import numpy as np

from utils.plots import plot_results, plot_uv_ratio, plot_xy_ratio


def my_round(x):
    try:
        return np.around(x, decimals=1)
    except:
        return x


def save_results(df, result_dir, skip):
    df = df.loc[df['epoch'] >= skip]
    df = df.groupby(by=['test_name', 'inst_cond', 'reg', 'group_acc'],
                    dropna=False)['value'].mean()
    df.apply(my_round).to_csv(
        os.path.join(result_dir, "final_results.csv"), index=False)


def results(result_dir="results", skip=44):
    result_path = os.path.join(result_dir, "results.csv")
    test_df = pd.read_csv(result_path)

    plot_results(test_df, result_dir, skip)
    plot_uv_ratio(test_df, result_dir, skip)
    plot_xy_ratio(test_df, result_dir, skip)


if __name__ == '__main__':
    fire.Fire(results)
