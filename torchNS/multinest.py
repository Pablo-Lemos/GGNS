import torch
from torchNS.nested_sampling import NestedSampler
from torchNS.param import Param, NSPoints

# Default floating point type
dtype = torch.float32

class MultiNest(NestedSampler):
    def __init__(self, loglike, params, nlive=50, tol=0.1, max_nsteps=10000, verbose=True,
                 eff=0.1, clustering=False, device=None):
        super().__init__(loglike, params, nlive, tol, max_nsteps, verbose=verbose, clustering=clustering, device=device)
        self.eff = eff

    def fit_normal(self):
        x = self.live_points.get_values()
        mean = torch.mean(x, dim=0)
        cov = (1/self.eff)*torch.cov(x.T)
        assert torch.linalg.det(cov) > 0, "Covariance not positive semidefinite"
        mvn = torch.distributions.MultivariateNormal(mean,
                                                     scale_tril=torch.linalg.cholesky(cov).to(self.device))
        return mvn

    def find_new_sample(self, min_like):
        ''' Sample the prior until finding a sample with higher likelihood than a
        given value
        Parameters
        ----------
          min_like : float
            The threshold log-likelihood
        Returns
        -------
          newsample : pd.DataFrame
            A new sample
        '''

        newlike = -torch.inf
        mvn = self.fit_normal()
        while newlike < min_like:
            values = mvn.sample()
            newlike = self.loglike(values)
            self.like_evals += 1

        sample = NSPoints(self.nparams)
        sample.add_samples(values=values.reshape(1, -1),
                           logL=newlike.reshape(1), #torch.tensor([newlike], dtype = dtype),
                           weights=torch.ones(1, device=self.device))

        return sample

if __name__ == "__main__":
    ndims = 3
    mvn1 = torch.distributions.MultivariateNormal(loc=10*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.5*torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-10*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.5*torch.ones(ndims)))

    def get_loglike(theta):
        lp = torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=-1, keepdim=False)
        return lp

    params = []

    for i in range(ndims):
        params.append(
            Param(
                name=f'p{i}',
                prior_type='Uniform',
                prior=(-50, 50),
                label=f'p_{i}')
        )

    ns = MultiNest(
        nlive=25*ndims,
        loglike=get_loglike,
        params=params)

    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 / 10.**ndims))
    print('Number of evaluations', ns.get_like_evals())

    from getdist import plots
    samples = ns.convert_to_getdist()
    g = plots.get_subplot_plotter()
    g.triangle_plot([samples], filled=True)
    g.export('test.png')
