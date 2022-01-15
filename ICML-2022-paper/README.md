# ICML 2022 Paper

The paper pdf can be found [here](main.pdf).

## 2022-01-15

### Introduction

- [x] Explain group-instance disentanglement to the extent that everyone agreees with this (including me).
- [x] Give background to this goal.
- [x] What is the problem with the current methods?
  - [x] Why should anyone care?
- [x] What am I proposing?

## 2022-01-10

- [x] Move paper from `.md` to `.tex`.
- [x] Re-write introduction.

### Evaluation

- [ ] I should display results as violin plots instead of tables, in order to show the distribution of scores visually.
- [ ] I should change the evaluation procedure to the one in [Bouchacourt2018Multilevel](https://api.semanticscholar.org/CorpusID:1209557). Thus, predict the ground-truth group factor from the group and instance latents. The error should be low for the former and high for the latter.
  - [ ] Describe the new evaluation procedure
- [ ] (If I have time) I should run a comparison on standard image disentanglement datasets (e.g. dSprites, MPI3D, etc).

## 2022-01-06

- [x] Use bold notation to denote a group. So instead of $x_{[1:K]}$ write $\textbf{x}$. Where we want to show the whole group apart from one, write $\textbf{x}_{-k}$.
- [x] Write introduction.
- [ ] If I want to talk about the relationship between the ground-truth variable and the inferred variable, simply define a new model $s$ which encompasses both of them.