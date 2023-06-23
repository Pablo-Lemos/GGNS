import torch
from gradNS.nested_sampling import NestedSampler
from gradNS.param import Param, NSPoints

# Default floating point type
dtype = torch.float64

class DynamicNestedSampler(NestedSampler):
    """
    This Nested Sampler uses Dynamic Nested Sampling to sample the posterior.
    """
    def __init__(self, loglike, params, nlive=50, tol=0.1, clustering=False, verbose=True, device=None):
        """ In this Nested Sampler, we start with a set of live points, and instead of killing one at a time, we
        kill half of them and replace them with new samples from the prior. This is done until the tolerance is reached.
        """
        super().__init__(loglike, params, nlive, tol, clustering, verbose, device)

    def move_one_step(self):
        """
        Move one step in the Dynamic Nested Sampling algorithm, by replacing half the samples with a new half

        If using clustering, we kill points and assign a label, then add new points with the same label.
        """
        if self.n_clusters == 1:
            n_points = self.nlive_ini//2
            for _ in range(n_points):
                sample = self.kill_point()

            logl = sample.get_logL().clone()
            labels = torch.bincount(torch.zeros(n_points, dtype=torch.int))
            self.add_point_batch(min_logL=logl, n_points=n_points, labels=labels)
        else:
            sample = self.kill_point()
            new_labels = torch.zeros(self.n_clusters, dtype=torch.int)
            cluster_volumes = torch.exp(self.summaries.get_logXp())
            num_points = self.live_points.count_labels()
            idx = torch.multinomial(cluster_volumes, 1)
            new_labels[idx] += 1
            while torch.min(num_points - new_labels) > 1:
                sample = self.kill_point()
                cluster_volumes = torch.exp(self.summaries.get_logXp())
                num_points = self.live_points.count_labels()
                idx = torch.multinomial(cluster_volumes, 1)
                new_labels[idx] += 1

            logl = sample.get_logL().clone()
            self.add_point_batch(min_logL=logl, n_points=torch.sum(new_labels).item(), labels=new_labels)

    def add_point_batch(self, min_logL, n_points, labels=None):
        """
        Add a new sample to the live points, and assign it a label

        Parameters
        ----------
        min_logL : float
            The minimum log likelihood of the new sample
        n_points : int
            The number of points to add
        labels : torch.Tensor
            The labels of the new points

        Returns
        -------
        """
        newsample = self.find_new_sample_batch(min_logL, n_points=n_points, labels=labels)
        assert torch.max(newsample.get_logL()) > min_logL, "New sample has lower likelihood than old one"
        self.n_accepted += n_points
        self.live_points.add_nspoint(newsample)

    def find_new_sample_batch(self, min_like, n_points, labels=None):
        """
        Run a for loop over find_new_sample

        Parameters
        ----------
        min_like : float
            The minimum log likelihood of the new sample
        n_points : int
            The number of points to add
        labels : torch.Tensor
            The labels of the new points

        Returns
        -------
        sample : NSPoints
            The new sample
        """
        sample = NSPoints(self.nparams)

        # In the base class, find the points "brute force"
        for _ in range(n_points):
            newsample = self.find_new_sample(min_like)
            sample.add_nspoint(newsample)

        return sample