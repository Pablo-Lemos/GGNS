import torch
from torchNS.nested_sampling import NestedSampler
from torchNS.param import Param, NSPoints

# Default floating point type
dtype = torch.float64


class SliceNS(NestedSampler):
    """
    This Nested Sampler uses Slice Sampling to generate new samples.
    """
    def __init__(self, loglike, params, nlive=50, tol=0.1, num_repeats=None, verbose=True, clustering=False, device=None):
        super().__init__(loglike, params, nlive, tol, verbose=verbose, clustering=clustering, device=device)
        """ 
        Parameters
        ----------
        loglike : function
            The log-likelihood function
        params : list
            A list of Param objects
        nlive : int
            The number of live points
        tol : float
            The tolerance for the stopping criterion
        verbose : bool
            Whether to print information
        num_repeats : int
            The number of repeats for the slice sampling algorithm
        clustering : bool
            Whether to use clustering
        device : torch.device
            The device to use        
        """

        self.n_repeats = 2 * int(self.nparams) if num_repeats is None else num_repeats

        # self.given_score = score is not None
        # if self.given_score:
        #     self.score = score

    def slice_sampling(self, log_slice_height, initial_x, step_size=1):
        """
        Slice sampling algorithm for PyTorch.

        Parameters
        ----------
        log_slice_height : float
            The height of the slice
        initial_x : torch.Tensor
            The initial point
        step_size : float
            The step size for the slice sampling algorithm

        Returns
        -------
        x : torch.Tensor
            The new sample
        """

        x = initial_x.clone()
        # Choose a random direction
        d = torch.randn(self.nparams, dtype=dtype, device=self.device)
        d = d / torch.linalg.norm(d)

        temp_random = torch.rand(1, dtype=dtype, device=self.device)
        w = 1
        # Set the initial interval
        L = x - temp_random * w * d
        R = x + (1 - temp_random) * w * d

        # Shrink the interval until it brackets the slice
        # For left and right
        self.like_evals += 2

        # TODO this only works for uniform priors where all parameters have the same limits
        while (self.loglike(L) > log_slice_height) and (torch.min(L) > torch.min(self._lower) and (torch.max(L) < torch.max(self._upper))):
            self.like_evals += 1
            L = L - d * w * step_size
        while (self.loglike(R) > log_slice_height) and (torch.max(R) < torch.max(self._upper) and (torch.min(R) > torch.min(self._lower))):
            self.like_evals += 1
            R = R + d * w * step_size

        # Sample a new x within the interval
        while True:
            x0Ld = torch.linalg.norm((L - x))
            x0Rd = torch.linalg.norm((R - x))

            new_x = x + (torch.rand(1, dtype=dtype, device=self.device) * (x0Rd + x0Ld) - x0Ld) * d
            new_log_prob, _ = self.get_score(new_x)

            if new_log_prob > log_slice_height:
                x = new_x.clone()
                break
            else:
                if (new_x - x).dot(d) < 0:
                    L = new_x.clone()
                else:
                    R = new_x.clone()

        return x, new_log_prob

    def find_new_sample(self, min_like):
        """
        Use slice sampling to find a new sample.

        Parameters
        ----------
        min_like : float
            The minimum likelihood value

        Returns
        -------
        sample : NSPoints
            The new sample
        """
        cluster_volumes = torch.exp(self.summaries.get_logXp())
        initial_point = self.live_points.get_random_sample(cluster_volumes)
        curr_value = initial_point.get_values()[0]

        for _ in range(self.n_repeats):
            new_value, new_loglike = self.slice_sampling(min_like, curr_value)
            curr_value = new_value.clone()

        sample = NSPoints(self.nparams)
        sample.add_samples(values=new_value.reshape(1, -1),
                           logL=new_loglike.reshape(1),
                           logweights=torch.zeros(1, device=self.device),
                           labels=initial_point.get_labels())

        return sample