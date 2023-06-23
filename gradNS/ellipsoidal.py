import torch
from gradNS.nested_sampling import NestedSampler
from gradNS.param import Param, NSPoints

# Default floating point type
dtype = torch.float64


class EllipsoidalNS(NestedSampler):
    """
    This Nested Sampler uses Ellipsoids to sample the posterior.
    """
    def __init__(self, loglike, params, nlive=50, tol=0.1, verbose=True, eff=0.1, clustering=False, device=None):
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
        eff : float
            The MultiNest efficiency, i.e. how much do we increase the ellipsoid sizes
        clustering : bool
            Whether to use clustering
        device : torch.device
            The device to use
        """
        super().__init__(loglike, params, nlive, tol, verbose=verbose, clustering=clustering, device=device)
        self.eff = eff
        if clustering:
            raise NotImplementedError("Clustering not implemented for MultiNest")

    def fit_normal(self):
        """
        Fit a multivariate normal to the live points
        Returns
        -------
        mvn : torch.distributions.MultivariateNormal
            The multivariate normal distribution
        """
        x = self.live_points.get_values()
        mean = torch.mean(x, dim=0)
        cov = (1/self.eff)*torch.cov(x.T)
        assert torch.linalg.det(cov) > 0, "Covariance not positive semidefinite"
        mvn = torch.distributions.MultivariateNormal(mean,
                                                     scale_tril=torch.linalg.cholesky(cov).to(self.device))
        return mvn

    def find_new_sample(self, min_like):
        """ Sample the prior until finding a sample with higher likelihood than a
        given value
        Parameters
        ----------
          min_like : float
            The threshold log-likelihood
        Returns
        -------
          sample : pd.DataFrame
            A new sample
        """
        newlike = -torch.inf
        mvn = self.fit_normal()
        while newlike < min_like:
            values = mvn.sample()
            newlike = self.loglike(values)
            self.like_evals += 1

        sample = NSPoints(self.nparams)
        sample.add_samples(values=values.reshape(1, -1),
                           logL=newlike.reshape(1), #torch.tensor([newlike], dtype = dtype),
                           logweights=torch.ones(1, device=self.device))

        return sample