import unittest
import torch
import numpy as np
from gradNS import Param, NestedSampler, EllipsoidalNS, SliceNS, DynamicNestedSampler, HamiltonianNS


def test_clustering(sampler='base', ndims=2):
    assert sampler in ['base', 'ellipsoidal', 'slice', 'dynamic', 'hamiltonian', 'dev'], 'Test not implemented for this ' \
                                                                                  'sampler'
    mvn1 = torch.distributions.MultivariateNormal(loc=-2 * torch.ones(ndims),
                                                  covariance_matrix=torch.diag(
                                                      0.2 * torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=2 * torch.ones(ndims),
                                                  covariance_matrix=torch.diag(
                                                      0.2 * torch.ones(ndims)))

    def get_loglike(theta):
        return torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=0, keepdim=False)

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
            clustering=True,
            verbose=False)
    elif sampler == 'ellipsoidal':
        ns = EllipsoidalNS(
            nlive=25 * ndims,
            loglike=get_loglike,
            params=params,
            eff=1.,
            clustering=True,
            verbose=False)
    elif sampler == 'slice':
        ns = SliceNS(
            nlive=25 * ndims,
            loglike=get_loglike,
            params=params,
            clustering=True,
            verbose=False)
    elif sampler == 'dynamic':
        ns = DynamicNestedSampler(
                    nlive=25 * ndims,
                    loglike=get_loglike,
                    params=params,
                    clustering=True,
                    verbose=False)
    elif sampler == 'hamiltonian':
        ns = HamiltonianNS(
            nlive=25 * ndims,
            loglike=get_loglike,
            params=params,
            clustering=True,
            verbose=False)

    # Run the sampler
    ns.run()
    return ns.get_mean_logZ(), ns.get_var_logZ()**0.5


class ClusteringTest(unittest.TestCase):
    def test_base(self):
        ndims = 2
        logZ, logZerr = test_clustering(sampler='base', ndims=ndims)
        self.assertAlmostEqual(logZ,
                               np.log(1 / 10 ** ndims),
                               delta=10*logZerr)

    def test_slice(self):
        ndims = 2
        logZ, logZerr = test_clustering(sampler='slice', ndims=ndims)
        self.assertAlmostEqual(logZ,
                               np.log(1 / 10 ** ndims),
                               delta=10*logZerr)

    def test_dynamic(self):
        ndims = 2
        logZ, logZerr = test_clustering(sampler='dynamic', ndims=ndims)
        self.assertAlmostEqual(logZ,
                               np.log(1 / 10 ** ndims),
                               delta=10*logZerr)

    def test_hamiltonian(self):
        ndims = 2
        logZ, logZerr = test_clustering(sampler='hamiltonian', ndims=ndims)
        self.assertAlmostEqual(logZ,
                               np.log(1 / 10 ** ndims),
                               delta=10*logZerr)

if __name__ == '__main__':
    unittest.main()
