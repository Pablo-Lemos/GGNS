import torch
from sklearn.mixture import GaussianMixture

def gmm_bic(X, max_components=None):
    curr_bic = 1e300
    n_components = 0
    max_components = X.shape[0] if max_components is None else max_components
    converged = False
    prev_gmm = None
    while not converged:
        n_components += 1
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X)
        new_bic = gmm.bic(X)
        converged = new_bic > curr_bic or n_components >= max_components
        curr_bic = new_bic.copy()
        prev_gmm = gmm
    return prev_gmm.n_components, prev_gmm.fit_predict(X)


def uniform(low, high, size, dtype):
    u = torch.rand(size, dtype = dtype)
    return u*(high-low)+low


@torch.jit.script
def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    From: https://github.com/pytorch/pytorch/issues/61292
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out

