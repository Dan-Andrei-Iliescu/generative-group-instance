import torch
import pyro.poutine as poutine


# ELBO loss for hierarchical variational autoencoder
def elbo_func(model, x):
    # run the guide and trace its execution
    guide_trace = poutine.trace(model.guide).get_trace(x)
    # run the model and replay it against the samples from the guide
    model_trace = poutine.trace(
        poutine.replay(model.model, trace=guide_trace)).get_trace(x)
    # construct the elbo loss function
    return -1 * (model_trace.log_prob_sum() - guide_trace.log_prob_sum())


def elbo_func(model, x):


def grad_penalty_func(params, loss):
    # Creates gradients
    grad_params = torch.autograd.grad(
        outputs=loss, inputs=params, create_graph=True)

    # Computes the penalty term
    grad_norm = 0
    for grad in grad_params:
        grad_norm += grad.pow(2).sum()
    grad_norm = grad_norm.sqrt()

    return grad_norm


def v_vs_n_func(model, x):
    # Get instance variables
    _, v = model.guide(x)

    # Disc loss for instance variables
    fake_d, _ = model.adv(v)

    # Disc loss for normal distribution
    gt = torch.normal(mean=torch.zeros_like(v), std=torch.ones_like(v))
    real_d, _ = model.adv(gt)

    return torch.mean(real_d - fake_d)
