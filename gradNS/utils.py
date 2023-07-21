import torch
from sklearn.mixture import GaussianMixture
from collections import deque

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
        converged = 1.5*new_bic > curr_bic or n_components >= max_components
        curr_bic = new_bic.copy()
        prev_gmm = gmm
    return prev_gmm.n_components, prev_gmm.fit_predict(X)

def knn(x, n_neighbors):
    # Compute distances from points to all other points
    distances = torch.cdist(x, x)
    # Get indices of k nearest neighbors for each point
    knn_indices = torch.topk(distances, k=n_neighbors+1, largest=False, dim=1).indices[:, 1:]
    return knn_indices

def cluster_points(x, n_neighbors):
    num_points = x.size(0)
    l = knn(x, n_neighbors)
    cluster_sizes = []
    point_clusters = torch.zeros(num_points, dtype=torch.int32)
    cluster_idx = 0
    to_visit = set(range(l.shape[0]))
    stack = deque()
    while len(to_visit) > 0:
        idx = to_visit.pop()
        cluster_size = 0
        stack.append(idx)
        while len(stack) > 0:
            curr_i = stack.popleft()
            cluster_size += 1
            point_clusters[curr_i] = cluster_idx
            unique_indices = torch.unique(l[curr_i])
            for i in unique_indices:
                if i.item() in to_visit:
                    to_visit.remove(i.item())
                    stack.append(i.item())
        cluster_sizes.append(cluster_size)
        cluster_idx += 1

    return cluster_sizes, point_clusters

def get_knn_clusters(x, max_components=None):
    max_components = x.shape[0] if max_components is None else max_components
    ns = [i for i in range(max_components)]
    sizes_prev = None
    for n_neighbors in ns:
        sizes, labels = cluster_points(x, n_neighbors=n_neighbors)
        if (sizes == sizes_prev) and (min(sizes) > 2 * x.shape[1]):
            #print(f"Stopping with {len(sizes)} clusters")
            break

        sizes_prev = sizes
    return len(sizes), labels

def uniform(low, high, size, dtype, device):
    u = torch.rand(size, dtype=dtype, device=device)
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

