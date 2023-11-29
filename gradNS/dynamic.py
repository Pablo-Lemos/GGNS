import torch
from gradNS.nested_sampling import NestedSampler
from gradNS.param import Param, NSPoints

# Default floating point type
dtype = torch.float64

class DynamicNestedSampler(NestedSampler):
    """
    This Nested Sampler uses Dynamic Nested Sampling to sample the posterior.
    """
    def __init__(self, loglike, params, nlive=50, tol=0.1, rejection_fraction=0.1, clustering=False, verbose=True,
                 device=None):
        """ In this Nested Sampler, we start with a set of live points, and instead of killing one at a time, we
        kill half of them and replace them with new samples from the prior. This is done until the tolerance is reached.
        """
        super().__init__(loglike, params, nlive, tol, clustering, verbose, device)
        self.n_accepted_pure_ns = 0
        self.frac_pure_ns = 1.
        self.min_frac_pure_ns = rejection_fraction

    def move_one_step(self):
        """
        Move one step in the Dynamic Nested Sampling algorithm, by replacing half the samples with a new half

        If using clustering, we kill points and assign a label, then add new points with the same label.
        """
        while self.frac_pure_ns > self.min_frac_pure_ns:
            sample = self.kill_point()
            self.add_point(min_logL=sample.get_logL())
            self.n_accepted_pure_ns += 1
            self.frac_pure_ns = self.n_accepted_pure_ns / self.n_tried

        if self.n_clusters == 1:
            n_points = self.nlive_ini//2
            for _ in range(n_points):
                sample = self.kill_point()

            logl = sample.get_logL().clone()
            labels = torch.bincount(torch.zeros(n_points, device=self.device, dtype=torch.int))
            self.add_point_batch(min_logL=logl, n_points=n_points, labels=labels)
        else:
            n_points = self.nlive_ini // 2
            for _ in range(n_points):
                sample = self.kill_point()

            cluster_volumes = torch.exp(self.summaries.get_logXp())
            idx = torch.multinomial(cluster_volumes, n_points, replacement=True)
            labels = torch.bincount(idx)

            logl = sample.get_logL().clone()
            self.add_point_batch(min_logL=logl, n_points=n_points, labels=labels)
            # sample = self.kill_point()
            # new_labels = torch.zeros(self.n_clusters, dtype=torch.int)
            # cluster_volumes = torch.exp(self.summaries.get_logXp())
            # num_points = self.live_points.count_labels()
            # idx = torch.multinomial(cluster_volumes, 1)
            # new_labels[idx] += 1
            # while torch.min(num_points - new_labels) > 1:
            #     sample = self.kill_point()
            #     cluster_volumes = torch.exp(self.summaries.get_logXp())
            #     num_points = self.live_points.count_labels()
            #     idx = torch.multinomial(cluster_volumes, 1)
            #     new_labels[idx] += 1
            #
            # logl = sample.get_logL().clone()
            # self.add_point_batch(min_logL=logl, n_points=torch.sum(new_labels).item(), labels=new_labels)

            # IDEA: Try finding half the points based on initial cluster volumes

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
        newsample.logL_birth = min_logL * torch.ones(n_points, dtype=dtype)
        self.xlogL = torch.cat((self.xlogL, min_logL + self.summaries.get_logX()))
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
        sample = NSPoints(self.nparams, device=self.device)

        # In the base class, find the points "brute force"
        for _ in range(n_points):
            newsample = self.find_new_sample(min_like)
            sample.add_nspoint(newsample)

        return sample
