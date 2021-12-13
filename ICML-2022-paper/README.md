# Entangled Grouped Data
*Dan Andrei Iliescu*


## Generative Group-Instance

The Group Variational Autoencoder ([Bouchacourt2018MultiLevelVA](https://api.semanticscholar.org/CorpusID:1209557), [Hosoya2019GroupbasedLO](https://api.semanticscholar.org/CorpusID:199466320)) is a family of models that use two latent variables to represent grouped data: one that captures the variation within groups, and one for the variation across groups.

Assume a dataset of the form $x_{[1:N,~1:K_i]} = \{x_{ik}\}_{i \in 1:N, ~k \in 1:K_i}$ where $N$ is the number of groups and $K_i$ is the number of observations in group $i$. GVAE defines a generative model that maps a $\mathcal{N} (0,1)$ group latent variable $u_i$ and a $\mathcal{N} (0,1)$ instance latent variable $v_{ik}$ to a given data observation $x_{ik}$. In other words, the likelihood of a group is:

$$p(x_{[1:K]}) = \mathbb{E}_{p(u)}  \prod_{k=1}^{K} \mathbb{E}_{p(v_{k})} ~ [p(x_{k} | u, v_{k})]$$

We omit the index of the group $i$ for notational simplicity, since the groups are independent and identically distributed.

### Variational Inference

Because the exact likelihood is intractable, the Variational Autodencoder ([Kingma2014AutoEncodingVB](https://api.semanticscholar.org/CorpusID:216078090), [JimenezRezende2014StochasticBA](https://api.semanticscholar.org/CorpusID:16895865)) performs optimization by introducting a variational latent posterior $q(u, v_{[1:K]} | x_{[1:K]})$ and maximizing the Evidence Lower Bound ([Jordan2004AnIT](https://api.semanticscholar.org/CorpusID:2073260)):

$$\log p(x_{[1:K]}) \geq \mathbb{E}_{q(u, v_{[1:K]} | x_{[1:K]})} [\log p(x_{[1:K]} | u, v_{[1:K]})] - \mathrm{KL} [q(u, v_{[1:K]} | x_{[1:K]}) || p(u, v_{[1:K]})]$$

The models in the GVAE family use a class of variational distributions that assume independence between the latent variables in a group when conditioned on the data.

$$q(u, v_{[1:K]} | x_{[1:K]}) = q(u | x_{[1:K]}) \prod_{k=1}^K q(v_k | x_k, u)$$

In this work, we show that this assumption hinders disentanglement when the generative model is entangled.

> **TODO:** Describe how the GVAE implements the Group Encoder and how it does regularization. I am proposing improvements to both of these elements.
> $$p(u | x_{[1:K]}) = \frac{\prod_{k=1}^K p(x_k)}{p(x_{[1:K]})} \frac{1}{p(u)^{K-1}} \prod_{k=1}^K p(u | x_k)$$

## How to Measure Disentanglement

In the context of the GVAE family, disentanglement is a property of the variational latent posterior. The inferred group and instance variables are disentangled when they are maximally informative about the group and instance variables of the true data generating distribution. We assume this true model has the same factorization of the joint distribution as the generative model, but the parameters are unknown.

For example, the 

$$\hat{q} = \argmax_q I(U^t; U^q)$$
$$= \argmax_q \mathrm{KL} [\mathrm{Pr}_{U^q, U^t} || \mathrm{Pr}_{U^q} \mathrm{Pr}_{U^t}]$$
$$= \argmax_q \mathbb{E}_{U^q} \mathbb{E}_{U^t | U^q} \left[ \log \frac{\mathrm{Pr}_{U^t | U^q}}{\mathrm{Pr}_{U^t}} \right]$$
and since $\mathrm{Pr}_{U^t}$ does not depend on $q$
$$= \argmax_q \mathbb{E}_{U^q} \mathbb{E}_{U^t | U^q} [ \log \mathrm{Pr}_{U^t | U^q}]$$

Unfortunately, the density $\mathrm{Pr} (U^t | U^q)$ is unknowable 

### Unsupervised Translation

> **TODO:** Because unsupervised translation is a common downstream task of disentanglement, we also use it as an evaluation metric.

## Entangled Group and Instance Variables

We call the group and instance variables *entangled* when they are not independent conditioned on the data $p(u, v | x) \neq p(u | x) p(v | x)$. A useful heuristic for establishing whether the variables are entangled is to ask "Does knowing the group variable for an observation influence my belief about its instance variable?"

This property of the generative model is present in many machine learning tasks, such as collaborative filtering, 3D novel view synthesis. For example, in the context of the [Netflix Challenge](https://en.wikipedia.org/wiki/Netflix_Prize), where the task was to predict what score a user would give to a new film, one cannot infer what film is associated with a given score without also knowing the user.

*Strictly speaking, most real-world models are entangled. However, in many cases, the mutual information between the group and instance variable, conditioned on the observation, is negligible. For example, in handwritten digit recognition, one can infer the digit value depicted in an image without knowing the author.*

In this paper, we claim that the current methods in the GVAE family do not perform well in tasks where the group and instance variables are entangled.

### Exam-Scores Problem

Suppose we wanted to model the exam scores of students from different schools. Our model must separate the school-level effect (the group factor) from the student-level effect (the instance factor). We define the following generative model:

$$x_{ik} = 2 ~ \mu_i +  (\sigma^2_i + 1) v_{ik} + \epsilon_{ik}, ~ i \in 1:N, ~ k \in 1:K_i$$

where $x_{ij}$ is the student score, $u_i = (\mu_i, \sigma_i)$ is the school-level effect, $v_{ij}$ is the student-level effect, and $\epsilon_{ik}$ is a normally-distributed error term. We assume a $\mathcal{N} (0, 1)$ prior distribution for the latent variables.

We first sample the model to generate a dataset ($N = 32,768, ~ K_i \sim \mathrm{Poisson} (16) + 8$) and then use the same model as the generative model in our Variational Autoencoder, instead of a neural network. The figure below shows what the data looks like.

<figure>
<img src="figures/data.svg">
<figcaption><b>Figure 2:</b> <i>Normalized exam scores of individual students grouped by school.</i></figcaption>
</figure>

Looking at the data, it is easy to see that this model is entangled, because the relative performance $v$ of a student within their own school, given their absolute score $x$, depends on the distribution of scores within the school $u$.

<figure>
<img src="figures/results.svg">
<figcaption><b>Figure 3:</b> <i>Performance metrics (as a function of the number of training epochs), comparing our method CxVAE (yellow, red) with the rest of the GVAE family (purple, magenta and blue). Our method outperforms the others according to every metric. <b>Left:</b> Reconstruction loss on holdout set. <b>Right:</b> Translation error on holdout set. </i></figcaption>
</figure>

### 

## Context-Aware Variational Autoencoder (CxVAE)

We propose a new model which can perform well on these tasks.

##

## Interpretation

We hypothesize that the current methods perform poorly due to the following 3 limitations:
1. The implementation of the group encoder does not correctly approximate the group latent posterior.
2. The form of the variational instance posterior is incorrect.
3. Use invariance to nuisance variables as regularization (instead of nemeth's method)

## Evaluation

We measure the loss 

| Model                                                                           | Rec Error          | Trans Error         | U Pred Error       | V Pred Error       |
| ------------------------------------------------------------------------------- | ------------------ | ------------------- | ------------------ | ------------------ |
| *CxVAE (ours)*                                                                  | **52.1** $\pm$ 0.6 | **219.5** $\pm$ 4.7 | **53.3** $\pm$ 1.2 | **30.2** $\pm$ 0.3 |
| [Bouchacourt2018MultiLevelVA](https://api.semanticscholar.org/CorpusID:1209557) | 69.8 $\pm$ 1.6     | 738.8 $\pm$ 13.6    | 63.4 $\pm$ 1.0     | 63.0 $\pm$ 0.3     |
| [Hosoya2019GroupbasedLO](https://api.semanticscholar.org/CorpusID:199466320)    | 71.9 $\pm$ 4.3     | 742.9 $\pm$ 14.4    | 63.1 $\pm$ 0.9     | 63.0 $\pm$ 0.4     |
| [Nmeth2020AdversarialDW](https://api.semanticscholar.org/CorpusID:210472540)    | 70.0 $\pm$ 1.4     | 735.3 $\pm$ 13.0    | 63.8 $\pm$ 1.1     | 63.0 $\pm$ 0.5     |


### Ablation Study

In order to quantify the effect of each proposed improvement, we perform an ablation study whereby we measure the decrease in performance resulting from replacing a proposed element of our model with a current alternative.

|                    | Rec Error          | Trans Error         | U Pred Error       | V Pred Error       |
| ------------------ | ------------------ | ------------------- | ------------------ | ------------------ |
| *CxVAE (ours)*     | **52.1** $\pm$ 0.6 | **219.5** $\pm$ 4.7 | **53.3** $\pm$ 1.2 | **30.2** $\pm$ 0.3 |
| **U Encoder**      |
| Average            | 53.6 $\pm$ 2.2     | 224.7 $\pm$ 10.0    | 53.4 $\pm$ 1.1     | 30.9 $\pm$ 1.1     |
| Multiplication     | 54.9 $\pm$ 2.8     | 229.2 $\pm$ 7.8     | 53.4 $\pm$ 0.9     | 31.4 $\pm$ 1.0     |
| **V Encoder**      |                    |                     |
| Unconditional      | 72.3 $\pm$ 3.6     | 729.0 $\pm$ 15.1    | 63.6 $\pm$ 1.2     | 63.0 $\pm$ 0.3     |
| **Regularization** |                    |                     |
| Unconditional IB   | 53.7 $\pm$ 1.2     | 225.7 $\pm$ 5.7     | 53.5 $\pm$ 1.1     | 30.8 $\pm$ 0.6     |
| None               | 54.1 $\pm$ 1.9     | 235.5 $\pm$ 16.5    | 53.6 $\pm$ 1.4     | 31.6 $\pm$ 1.5     |


### Ablation Study

In order to quantify the effect of each proposed improvement, we run the same experiment for every possible combination of options for the 3 elements: group encoder, instance encoder and regularization. This yields $3 \times 2 \times 3 = 18$ test conditions. Then, for each option of each element, we display the average performance over the options of the other two elements (e.g. the error for the "DeepSet" group encoder is the mean of the errors for the 6 test conditions that have "DeepSet" group encoders). The results can be seen in the table below.

|                         | Reconstruction Error | Translation Error    |
| ----------------------- | -------------------- | -------------------- |
| **Group Encoder**       |                      |
| *DeepSet (ours)*        | **51.3** $\pm$ 2.2   | **374.6** $\pm$ 12.3 |
| Average                 | 51.4 $\pm$ 2.2       | 374.9 $\pm$ 11.4     |
| Multiplication          | 55.0 $\pm$ 3.9       | 378.2 $\pm$ 11.7     |
| **Instance Encoder**    |                      |                      |
| *Group-aware (ours)*    | **33.5** $\pm$ 1.6   | **351.1** $\pm$ 13.2 |
| Unconditional           | 71.6 $\pm$ 3.9       | 400.7 $\pm$ 10.4     |
| **Regularization**      |                      |                      |
| *Group-aware IB (ours)* | **51.2** $\pm$ 3.9   | **373.5** $\pm$ 10.7 |
| Unconditional IB        | 51.6 $\pm$ 2.3       | 376.0 $\pm$ 12.5     |
| None                    | 54.8 $\pm$ 2.2       | 378.2 $\pm$ 12.2     |

The proposed improvements have the best individual performance with respect to both reconstruction and translation error in each of the three categories. The choice of a group-aware instance encoder leads to the most significant increase in performance.