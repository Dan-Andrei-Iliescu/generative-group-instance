import glob
import fire
import json
import os

from utils.plots import violin_plot as plot
from utils.helpers import save_results


def results(result_dir="results", palette=None):
    file_list = sorted(glob.glob(os.path.join(result_dir, "*")))
    test_dict = {}
    for file_path in file_list:
        # Record model name
        model_name = file_path.split("/")[-1]

        # Read test dict
        try:
            fileObject = open(file_path, "r")
            jsonContent = fileObject.read()
            test_dict[model_name] = json.loads(jsonContent)
        except:
            print("Not json")

    plot(test_dict, result_dir, palette)
    save_results(test_dict, result_dir)


if __name__ == '__main__':
    fire.Fire(results)
