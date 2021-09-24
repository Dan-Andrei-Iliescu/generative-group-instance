import glob
import fire
import json
import os
from utils.plots import plot_results


def results(result_dir="results"):
    file_list = sorted(glob.glob(os.path.join(result_dir, "*")))
    test_dict = {}
    for file_path in file_list:
        # Record model name
        model_name = file_path.split("/")[-1]

        # Read test dict
        fileObject = open(file_path, "r")
        jsonContent = fileObject.read()
        test_dict[model_name] = json.loads(jsonContent)

    plot_results(test_dict)


if __name__ == '__main__':
    fire.Fire(results)
