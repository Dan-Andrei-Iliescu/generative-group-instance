# Disentangling Highly Entangled Grouped Data

**DEF:** We call a generative model highly entangled when the group and instance variables are not independent given the data $p(u, v | x) \neq p(u | x) p(v | x)$.

This rule property characterizes many ML tasks to which Group-Instance disentanglement doesn't currently produce good results (**e.g.** federated learning, recommender systems, 3D novel view synthesis).

We propose a new model which can perform well on these tasks.

We hypothesize that the current methods work poorly due to the following 3 limitations:
1. The implementation of the group encoder does not correctly approximate the group latent posterior.
2. The form of the variational instance posterior is incorrect.
3. What regularization is used suffers from the same problem as the variational instance posterior.