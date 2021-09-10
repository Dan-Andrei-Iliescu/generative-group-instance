# generative-group-instance

Disentangle group-specific variables from group-independent variables using Variational Autoencoders.

This repository contains multiple learning tasks/datasets, corresponding to the directories in the root (e.g. river-flow, exam-scores etc). **Every script should be run at the top level of the directory corresponding to that task/dataset.** For example, if you want to run `generative-group-instance/dir1/dir2/dir3/script.py`, you should run:

```
generative-group-instance/dir1$ python dir2/dir3/script.py
```
This is in order to have consistent relative paths in the python script.

## Poetry

[Poetry](https://python-poetry.org) is a dependency and package management software for Python. It is meant to replace anaconda and pip. It creates virtual environments for Python on-the-fly with dependencies specified in files called `pyproject.toml`. 

In this repository, each learning task/dataset has its own Poetry virtual environment, specified by a separate `pyproject.toml` and `poetry.lock` file for each directory in the root folder. You can add new packages to the virtual environment or use it to run Python scripts. Find more information on how to install and use Poetry [here](https://python-poetry.org/docs/).

Some basic commands:
* `$ poetry init` - Initialize poetry in a new Python project.
* `$ poetry add [package_name]` - Add a package to the virtual environment of this project.
* `$ poetry remove [package_name]` - Remove a package to the virtual environment.
* `$ poetry run python [script_name].py` - Run a python script with the virtual environment.