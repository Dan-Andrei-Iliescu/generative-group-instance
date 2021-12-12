# exam-scores dataset

This is a synthetic dataset of exam scores.

## How to run

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

## Idea: Provide the generative model, learn only the variational model

So far in the implementation, the model is required to learn both the decoder and the encoder. However, both the correct model and the wrong model achieve poor translation quality, because the order of the samples is scrambled when changing the group variable. I hypothesize that the reason for this is too many degrees of freedom in the decoder.

I will try implementing the model with a fixed decoder matching the data generative process precisely. Maybe this restriction will allow the correct model to translate correctly, while the wrong model still fails.