### 2021-12-09

I am claiming that disentanglement means maximizing the mutual information between the inferred latents and their "true" counterparts. In order to measure this mutual information empirically, I am training a network to predict the true factor from the inferred latent (and also maybe a network to predict the other factor from the inferred latent). The accuracy of this predictor on the testing set will be the measure of disentanglement.

In order to implement this experiment, I have to change the data generating procedure to also record the ground truth factors. Additionally, I have to implement the new network and add it to the training loop.