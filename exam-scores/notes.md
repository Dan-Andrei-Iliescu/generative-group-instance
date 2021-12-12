### 2021-12-12

I have also changed the generation procedure. I think a problem with the disentanglement was that, in some cases, the variance of the observations within a group was too small because the variance was a squared normal. I have added 1 to the variance term in the group variable, and disentanglement seems to have improved vastly.

### 2021-12-09

I am claiming that disentanglement means maximizing the mutual information between the inferred latents and their "true" counterparts. In order to measure this mutual information empirically, I am training a network to predict the true factor from the inferred latent (and also maybe a network to predict the other factor from the inferred latent). The accuracy of this predictor on the testing set will be the measure of disentanglement.

In order to implement this experiment, I have to 
- [x] Change the data generating procedure to also record the ground truth factors 
- [x] Implement the new predictor network
- [x] Implement the training and testing procedure
- [x] Record these predictions instead of the latent statistics

I've just realized there's a problem with my generation procedure. The generative prior for the scale of the group variable is a normal with mean 0. However, the scale must be positive.

The predictor network is a simple linear mapping.