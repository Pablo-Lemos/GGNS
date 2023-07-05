import unittest
import torch
import numpy as np
from gradNS import Param, Prior, NestedSampler, EllipsoidalNS, SliceNS, DynamicNestedSampler, HamiltonianNS


def test_clustering(sampler='base', ndims=2):
    assert sampler in ['base', 'hamiltonian'], 'Test not implemented for this ' \
                                                                                  'sampler'
    mvn1 = torch.distributions.MultivariateNormal(loc=-2 * torch.ones(ndims, dtype=torch.float64),
                                                  covariance_matrix=torch.diag(
                                                      0.2 * torch.ones(ndims, dtype=torch.float64)))

    mvn2 = torch.distributions.MultivariateNormal(loc=2 * torch.ones(ndims, dtype=torch.float64),
                                                  covariance_matrix=torch.diag(
                                                      0.2 * torch.ones(ndims, dtype=torch.float64)))

    mvn_like = torch.distributions.MultivariateNormal(loc=torch.zeros(ndims, dtype=torch.float64),
                                                  covariance_matrix=torch.diag(
                                                      0.2 * torch.ones(ndims, dtype=torch.float64)))

    def get_logprior(theta):
        return torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=0, keepdim=False) - torch.log(torch.tensor(2.0))

    def sample_prior(nsamples):
        # Generate samples from the prior
        # First generate a tensor of zeros and ones with equal probability
        mask = torch.randint(0, 2, (nsamples,))
        # Then use the mask to select the appropriate
        counts = torch.bincount(mask, minlength=2)
        samples = torch.cat([mvn1.sample((counts[0],)), mvn2.sample((counts[1],))], dim=0)
        # Shuffle the samples
        return samples[torch.randperm(nsamples)]

    def get_loglike(theta):
        return mvn_like.log_prob(theta)

    def get_prior_score(theta):
        theta = theta.clone().detach().requires_grad_(True)
        logprior = get_logprior(theta)
        return torch.autograd.grad(logprior, theta, torch.ones_like(logprior))[0]

    prior = Prior(score=get_prior_score, sample=sample_prior)

    params = [Param(name=f'p{i}',
                    prior_type='Uniform',
                    prior=(-5, 5),
                    label=f'p_{i}')
              for i in range(ndims)]

    if sampler == 'base':
        ns = NestedSampler(
            nlive=25 * ndims,
            loglike=get_loglike,
            params=params,
            tol=1., # Base is veru slow in this example
            clustering=False,
            verbose=False)
    elif sampler == 'hamiltonian':
        ns = HamiltonianNS(
            nlive=25 * ndims,
            loglike=get_loglike,
            params=params,
            clustering=True,
            verbose=False)

    ns.add_prior(prior)

    # Run the sampler
    ns.run()
    return ns.get_mean_logZ(), ns.get_var_logZ()**0.5


class ClusteringTest(unittest.TestCase):
    # The true logZ is -11 for 2 dimensions (from experiments)
    def test_base(self):
        ndims = 2
        logZ, logZerr = test_clustering(sampler='base', ndims=ndims)
        self.assertAlmostEqual(logZ,
                               -11.,
                               delta=10*logZerr)

    def test_hamiltonian(self):
        ndims = 2
        logZ, logZerr = test_clustering(sampler='hamiltonian', ndims=ndims)
        self.assertAlmostEqual(logZ,
                               -11.,
                               delta=10*logZerr)


if __name__ == '__main__':
    unittest.main()
