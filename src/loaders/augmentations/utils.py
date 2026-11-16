import torch

def mixup(x1, x2, alpha=0.4):
    '''Returns the mixup of x1 and x2 with mixup ratio sampled from Beta distribution with parameter alpha.
    :param x1: Tensor of shape (..., feature_dim)
    :type x1: torch.Tensor
    :param x2: Tensor of shape (..., feature_dim)
    :type x2: torch.Tensor
    :param alpha: Parameter for Beta distribution to sample mixup ratio. Default is 1.0.
    :type alpha: float, optional
    :return: The mixup of x1 and x2
    :rtype: torch.Tensor of shape (..., feature_dim)
    '''
    sample_dist = torch.distributions.Beta(alpha, alpha)
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape for mixup"
    if alpha > 0:
        lam = sample_dist.sample(x1.shape[:-1]).to(x1.device)
        lam = lam.unsqueeze(-1)  # shape (..., 1)
        return lam * x1 + (1 - lam) * x2
    else:
        return x1
