import unittest
import torch
import numpy as np
from torchNS import Param, NestedSampler, EllipsoidalNS

def test_nested_sampling(self, sampler='base', ndims=2):
    assert sampler in ['base', 'ellipsoidal'], 'Test not implemented for this sampler'
    mvn = torch.distributions.MultivariateNormal(loc=2 * torch.ones(ndims),
                                                 covariance_matrix=torch.diag(0.2 * torch.ones(ndims))
                                                 )

    params = [Param(name=f'p{i}',
                    prior_type='Uniform',
                    prior=(-5, 5),
                    label=f'p_{i}')
              for i in range(ndims)]

    if sampler == 'base':
        ns = NestedSampler(
            nlive=25 * ndims,
            loglike=mvn.log_prob,
            params=params,
            clustering=False,
            verbose=False)
    elif sampler == 'ellipsoidal':
        ns = EllipsoidalNS(
            nlive=25 * ndims,
            loglike=mvn.log_prob,
            params=params,
            eff=1.,
            clustering=False,
            verbose=False)

    # Run the sampler
    ns.run()
    return ns.get_mean_logZ(), ns.get_var_logZ()**0.5


class NestedSamplingTest(unittest.TestCase):
    def test_base(self):
        ndims = 2
        logZ, logZerr = test_nested_sampling(self, sampler='base', ndims=ndims)
        # Check that logZ is within 10 sigma of the true value
        self.assertAlmostEqual(logZ,
                               np.log(1 / 10 ** ndims),
                               delta=10*logZerr)

    def test_ellipsoidal(self):
        ndims = 2
        logZ, logZerr = test_nested_sampling(self, sampler='ellipsoidal', ndims=ndims)
        # Check that logZ is within 10 sigma of the true value
        self.assertAlmostEqual(logZ,
                               np.log(1 / 10 ** ndims),
                               delta=10*logZerr)


if __name__ == '__main__':
    unittest.main()
