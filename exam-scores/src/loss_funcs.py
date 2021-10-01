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


def grad_penalty_func(params, loss):
    # Creates gradients
    grad_params = torch.autograd.grad(
        outputs=loss, inputs=params, create_graph=True, allow_unused=True)

    # Computes the penalty term
    grad_norm = 0
    for grad in grad_params:
        if grad is not None:
            grad_norm += grad.pow(2).sum()
    grad_norm = grad_norm.sqrt()

    # print(f"Grad norm {grad_norm}")
    return grad_norm


def v_vs_n_func(model, x):
    # Get instance variables
    _, v = model.inference(x)

    # Disc loss for instance variables
    fake_d, _ = model.adv(v)

    # Disc loss for normal distribution
    gt = torch.normal(mean=torch.zeros_like(v), std=torch.ones_like(v))
    real_d, _ = model.adv(gt)

    loss = torch.sum(real_d - fake_d)

    # print(f"Adv loss {loss}")
    return loss


def nemeth_func(model, x):
    # Permutation on batch and group dimensions
    batch_size = x.shape[0]
    batch_order = torch.randperm(batch_size)
    group_size = x.shape[1]
    group_order = torch.randperm(group_size)

    # Inst vars
    _, v = model.inference(x)
    v_same = v[:, group_order]
    v_other = v[batch_order, group_order]

    # Adversarial loss
    d_same = model.adv(x, v_same)
    d_other = model.adv(x, v_other)

    loss = torch.sum(d_same) - torch.log(torch.sum(torch.exp(d_other)))

    return loss
