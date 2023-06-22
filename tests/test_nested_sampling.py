import unittest
import torch
from torchNS import NestedSampler, Param

class NestedSamplingTest(unittest.TestCase):
    def test_something(self, ndims=2):
        mvn = torch.distributions.MultivariateNormal(loc=2 * torch.ones(ndims),
                                                     covariance_matrix=torch.diag(0.2 * torch.ones(ndims))
                                                     )

        params = [Param(name=f'p{i}',
                       prior_type='Uniform',
                       prior=(-5, 5),
                       label=f'p_{i}')
                  for i in range(ndims)]

        ns = NestedSampler(
            nlive=25 * ndims,
            loglike=mvn.log_prob,
            params=params,
            clustering=False,
            verbose=False)

        ns.run()

        # The true logZ is the inverse of the prior volume
        import numpy as np
        print('True logZ = ', np.log(1 / 10 ** len(params)))
        print('Number of evaluations', ns.get_like_evals())

        self.assertAlmostEqual(ns.get_mean_logZ(),
                               np.log(1 / 10 ** len(params)),
                               delta=10*ns.get_var_logZ()**0.5)


if __name__ == '__main__':
    unittest.main()
