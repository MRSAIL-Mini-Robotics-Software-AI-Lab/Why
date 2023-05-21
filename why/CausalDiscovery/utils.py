"""
Helper functions for causal discovery
"""
import torch

# pylint: disable=invalid-name
def MMDLoss(x, y):
    """
    Maximum Mean Discrepancy Loss

    Parameters
    ----------
    x : torch.tensor
        Tensor of variables, shape = (batch_size, num_vars)
    y : torch.tensor
        Tensor of variables, shape = (batch_size, num_vars)

    Returns
    -------
    torch.tensor, shape=1
        MMD loss
    """
    device = x.device
    x1 = x.reshape(1, x.shape[0], x.shape[1])
    x2 = x.reshape(x.shape[0], 1, x.shape[1])
    diffxx = x1 - x2
    diffxx = torch.sum(diffxx * diffxx, dim=2)

    y1 = y.reshape(1, y.shape[0], y.shape[1])
    y2 = y.reshape(y.shape[0], 1, y.shape[1])
    diffyy = y1 - y2
    diffyy = torch.sum(diffyy * diffyy, dim=2)

    diffxy = x1 - y2
    diffxy = torch.sum(diffxy * diffxy, dim=2)

    XX = torch.zeros(diffxx.shape).to(device)
    YY = torch.zeros(diffxx.shape).to(device)
    XY = torch.zeros(diffxx.shape).to(device)
    bandwidth_range = [0.005, 0.05, 0.25, 0.5, 1, 5, 50]
    for bandwidth in bandwidth_range:
        XX += torch.exp(-diffxx * bandwidth)
        YY += torch.exp(-diffyy * bandwidth)
        XY += torch.exp(-diffxy * bandwidth)

    return torch.mean(XX + YY - 2 * XY)
