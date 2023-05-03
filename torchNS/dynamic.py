import torch
from torchNS.nested_sampling import NestedSampler
from torchNS.param import Param, NSPoints

# Default floating point type
dtype = torch.float64

class DynamicNestedSampler(NestedSampler):
    def __init__(self, loglike, params, nlive=50, tol=0.1, max_nsteps=10000, verbose=True,
                 clustering=False, device=None):
        super().__init__(loglike, params, nlive, tol, max_nsteps, verbose=verbose, clustering=clustering, device=device)

    def move_one_step(self):
        ''' Find highest log likes, get rid of those point, and sample a new ones '''
        n_points = self.nlive_ini//2
        for i in range(n_points):
            sample = self.kill_point()

        self.add_point_batch(min_logL=sample.get_logL(), n_points=n_points)

    def add_point_batch(self, min_logL, n_points):
        # Add a new sample
        newsample = self.find_new_sample_batch(min_logL, n_points=n_points)
        assert torch.max(newsample.get_logL()) > min_logL, "New sample has lower likelihood than old one"

        if self.clustering:
            # Find closest point to new sample
            values = self.live_points.get_values()

            #TODO: Check if this is working for a batch
            dist = torch.sum((values - newsample.get_values()) ** 2, dim=1)
            idx = torch.argmin(dist)

            # Assign its label to the new point
            newsample.set_labels(self.live_points.get_labels()[idx].reshape(1))
        self.live_points.add_nspoint(newsample)

    def find_new_sample_batch(self, min_like, n_points):
            ''' Run a for loop over find_new_sample
            '''
            sample = NSPoints(self.nparams)

            for i in range(n_points):
                newsample = self.find_new_sample(min_like)
                sample.add_nspoint(newsample)

            return sample


if __name__ == "__main__":
    ndims = 2
    mvn1 = torch.distributions.MultivariateNormal(loc=2 * torch.ones(ndims),
                                                  covariance_matrix=torch.diag(
                                                      0.2 * torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-1 * torch.ones(ndims),
                                                  covariance_matrix=torch.diag(
                                                      0.2 * torch.ones(ndims)))

    true_samples = torch.cat([mvn1.sample((5000,)), mvn2.sample((5000,))], dim=0)

    def get_loglike(theta):
        lp = torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=-1,
                             keepdim=False) - torch.log(torch.tensor(2.0))
        return lp


    params = []

    for i in range(ndims):
        params.append(
            Param(
                name=f'p{i}',
                prior_type='Uniform',
                prior=(-5, 5),
                label=f'p_{i}')
        )

    ns = DynamicNestedSampler(
        nlive=25*ndims,
        loglike=get_loglike,
        params=params,
        clustering=True,
        verbose=True,)

    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 / 10**len(params)))
    print('Number of evaluations', ns.get_like_evals())

    from getdist import plots, MCSamples
    samples = ns.convert_to_getdist()
    true_samples = MCSamples(samples=true_samples.numpy(), names=[f'p{i}' for i in range(ndims)])
    g = plots.get_subplot_plotter()
    g.triangle_plot([true_samples, samples], filled=True, legend_labels=['True', 'GDNest'])
    g.export('test.png')