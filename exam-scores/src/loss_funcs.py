import torch


# ELBO loss for hierarchical variational autoencoder
def elbo_func(model, x):
    # Latent inference dists
    qu, qv = model.inference(x, sample=False)

    # Sample from latent inference dists
    u = qu.rsample()
    v = qv.rsample()

    # Latent prior dists
    pu, pv = model.prior(u, v, sample=False)

    # Generative data dist
    x_loc = model.decoder.forward(u, v)

    # Losses
    kl_u = torch.sum(qu.log_prob(u) - pu.log_prob(u))
    # print(f"KL u {kl_u}")
    kl_v = torch.sum(qv.log_prob(v) - pv.log_prob(v))
    # print(f"KL v {kl_v}")
    lik = torch.sum((x - x_loc)**2)
    # print(f"lik {lik}")
    loss = kl_u + kl_v + lik

    # print(f"ELBO loss {loss}")
    return loss
