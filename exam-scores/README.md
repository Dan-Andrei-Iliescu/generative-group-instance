# exam-scores dataset

This is a synthetic dataset of exam scores.

In order to train the models for an experiment, run:

```
$ poetry run python src/experiments.py --exp_name model_type
```

In order to get the results without re-training, run

```
$ poetry run python src/experiment.py --training False
```

The results should be in the directories like `results/model_type`. There should be `.json` files with detailed quantitative results, `.svg` plots with qualitative and quantitaitve results, and `.csv` tables with summarized quantitative results.

### !Beware!

Because of a fault with `Poetry` and `Plotly`, you need to install kaleido (for saving `Plotly` figures as files) manually every time you make a change to the `Poetry` files or virtual environment. Run this sequence of commands:

```
$ poetry shell
$ which python # Check that you are in the right virutalenv
$ pip install -U kaleido
$ deactivate
```