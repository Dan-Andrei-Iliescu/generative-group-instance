import glob
import fire
import json
import os
import numpy as np

from utils.plots import plot_results


def compute_results(test_dict, result_dir):
    cond_names = list(test_dict.keys())
    for cond_name in cond_names:
        test_names = list(test_dict[cond_name].keys())
        for test_name in test_names:
            epochs = list(test_dict[cond_name][test_name].keys())
            error_mean = []
            error_sdev = []
            for epoch in epochs:
                runs = test_dict[cond_name][test_name][epoch]
                run_means = [np.mean(run) for run in runs]
                error_mean.append(np.mean(run_means))
                error_sdev.append(np.sqrt(np.mean(
                    (np.mean(run_means) - run_means)**2)))

            n = 5
            error_mean = np.mean(np.array(error_mean)[-n:])
            error_sdev = np.mean(np.array(error_sdev)[-n:])

            test_dict[cond_name][test_name] = {}
            test_dict[cond_name][test_name]['mean'] = error_mean
            test_dict[cond_name][test_name]['sdev'] = error_sdev

    # Save json file of test results
    save_path = os.path.join(result_dir, "table_results.json")
    jsonString = json.dumps(test_dict)
    with open(save_path, "w") as jsonFile:
        jsonFile.write(jsonString)


def results(result_dir="results"):
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

    plot_results(test_dict, result_dir)
    compute_results(test_dict, "figures")


if __name__ == '__main__':
    fire.Fire(results)
