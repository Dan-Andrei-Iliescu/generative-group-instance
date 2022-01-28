# ICML 2022 Paper

The paper pdf can be found [here](main.pdf).

## 2022-01-26

(6 hours)

## 2022-01-25

(6 hours) I have 12 more hours

1. Finish paper with new angle on contribution but without any extra datasets
   1. [x] Finish introduction with collaborative filtering example (0.5 hrs)
   2. [x] Update description of dataset (0.5 hrs)
   3. [ ] Update disentanglement metrics to show how they are rooted in the literature (0.5 hrs)
   4. [ ] Write related work section
      1. [ ] Talk about the area of weakly-supervised disentanglement, and what is different about it (1 hrs)
      2. [ ] Talk about how people deal with confounding in other areas, like mixed-models, collaborative filtering, causal inference, time series (1 hrs)
   5. [ ] Re-write experiments to highlight contribution (0.5 hrs)
   6. [ ] Write conclusions
      1. [ ] Summary of contributions
      2. [ ] Interpretation
   7. [x] Other tasks (1 hrs)


If I had an unlimited amount of time to make the paper perfect, what would I do?
1. I would test on collaborative filtering through multiple imputation using translation
2. I would test on multiple imputation in clinical data with their model
3. I would ascertain the level of disentanglement using the disentanglement metrics
4. I would run tests on real data to show my model is not worse than the rest
5. I would come up with a convincing definition of confounding and show that the datasets I am working with are confounded in this way.

## 2022-01-24

(4 hours)

1. Take out other two contributions, leave just instance conditioning
2. Finish paper
3. Add background paragraphs to each section placing it in the disentanglement literature
4. Give some context about how people deal with confounding
5. Compare with disentanglement datasets and use their metrics

I have 16 more hours.

- [x] Write background (0.5 hrs)
- [x] Model description only for instance conditioning. (0.5 hrs)
- [x] Found task and datasets (3 hrs)
  - Collaborative filtering on movielens dataset (easiest to implement and test)
  - Soybean (easiest to explain)
  - Alzheimers (most interesting and useful)
  - Pulmonary disese (also used in the disentanglement literature)
  - rotated MNIST
- [ ] Write introduction (1 hrs)
- [x] Updated paper to have just one contribution (0.5 hrs)

### Datasets + Tasks
- Collaborative filtering on MovieLens dataset
- 

## 2022-01-23

## Code
- [x] New experiment where I vary the relationship between the confounded component of the group and the unconfounded one.
- [x] Adjust relative strength experiment to have the decoder the same as the data-generating process.
- [x] Redo plots for ablation study.
- [ ] Run final experiments with more seeds.

## 2022-01-21

**Title:** One Datapoint is Not Enough: Disentangling Grouped Data with Confounding

**Abstract:** Group-instance disentanglement is the problem of mapping grouped data to separate representations for within-group and across-group variation. We introduce the Context-Aware Variational Autoencoder (CxVAE), a method that can perform group-instance disentanglement in data with confounding (i.e. where a single observation is not sufficient to accurately infer the group and instance variables). First, we generate a dataset with confounding which cannot be disentangled by the current state-of-the-art methods. Next, we improve upon these methods by proposing 3 modifications: 1) conditioning the instance variable on the group variable, 2) a more expressive group encoder, 3) a regularization objective that encourages independence between the instance variable and the grouping. Our method considerably improves disentanglement performance measured by several metrics: holdout reconstruction error, unsupervised translation error, and latent code probing. Finally, we reveal how adjusting the parameters of the data-generating process affects the performance gap between our model and the state-of-the-art.

## 2022-01-19

### Background

- [ ] Differentiate between the common ground, their approach, and my contributions.
  - [ ] The common ground is the Group-Instance Generative Model (GIGM). Additionally, we use the VAE paradigm to train this model.
  - [ ] The decisions of individual models revolve around using different variational models and regularization objectives. Why is this not just a straightforward application of the VAE paradigm? Difficulties are: how to factorize the distribution, how to accumulate evidence, how to prevent bleeding of information into one variable from the wrong factor.

### CxVAE

- [ ] Justify more my choices.
  - [ ] The instance conditioning follows the correct generative posterior.
  - [ ] The regularization mimics the above insight.
  - [ ] The group encoder makes no assumptions about the data.

### Related work

- [ ] Write section
  - [ ] Weakly-supervised disentanglement is different because they are not accumulating evidence.
  - [ ] GI is a subclass of mixed models. However, mixed models are useful for providing confidence intervals for fixed effects, whereas GI are used for translation. It is clear that GI methods are much more expressive and can deal with nonlinearities.
 
## 2022-01-17

### Introduction

- [x] State the goal of disentanglement in terms of some notation.
- [x] Take out VAE diagram and clarify in the caption what the reader should notice.
- [x] Restate the goal of disentanglement in terms of the GVAE. Why is this difficult, and not a straightforward application of the VAE?
- [x] Schools and exam scores is not a substantive analogy
- [x] Restate the necessity for conditioning the instance variable on the group variable.

### Evaluation

- [x] Re-write description of metrics
- [x] Re-write captions for figures.
- [x] Explain results
  - [x] Straight results
  - [x] Ablation study
  - [x] UV ratio
  - [x] XY ratio
- [x] I should make another experiment in which I vary the impact of the group variable in relation to the instance variable. If the group variable is too great, then

### Code

- [x] Restructure evaluation figures
- [x] Move test records to pandas
- [x] Reformat all figures
- [x] Fix ablation study

## 2022-01-15

### Introduction

- [x] Explain group-instance disentanglement to the extent that everyone agrees with this (including me).
- [x] Give background to this goal.
- [x] What is the problem with the current methods?
  - [x] Why should anyone care?
- [x] What am I proposing?

## 2022-01-10

- [x] Move paper from `.md` to `.tex`.
- [x] Re-write introduction.

### Evaluation

- [x] I should display results as violin plots instead of tables, in order to show the distribution of scores visually.
- [x] I should change the evaluation procedure to the one in [Bouchacourt2018Multilevel](https://api.semanticscholar.org/CorpusID:1209557). Thus, predict the ground-truth group factor from the group and instance latents. The error should be low for the former and high for the latter.
  - [x] Describe the new evaluation procedure.

## 2022-01-06

- [x] Use bold notation to denote a group. So instead of $x_{[1:K]}$ write $\textbf{x}$. Where we want to show the whole group apart from one, write $\textbf{x}_{-k}$.
- [x] Write introduction.