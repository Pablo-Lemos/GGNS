import torch
from torchNS.nested_sampling import NestedSampler
from torchNS.param import Param, NSPoints
import numpy as np
import tqdm

import pyro.distributions as dist
import pyro.distributions.transforms as T

# Default floating point type
dtype = torch.float32


class LangevinNest(NestedSampler):
    def __init__(self, loglike, params, nlive=50, tol=0.1, max_nsteps=1000000, verbose=True, score=None, device=None):
        super().__init__(loglike, params, nlive, tol, max_nsteps, verbose, device)

        self.acceptance_rate = 1.0
        self.n_tries = 0
        self.n_accepted = 0
        self.n_repeats = 5 #int(steps_per_dim * self.nparams)

        self.given_score = score is not None
        if self.given_score:
            self.score = score

        self._lower = torch.tensor([p.prior[0] for p in self.params], dtype=dtype, device=self.device)
        self._upper = torch.tensor([p.prior[1] for p in self.params], dtype=dtype, device=self.device)
        #assert p.prior_type == "uniform" for p in self.params, "Prior must be uniform for now"

    def get_score(self, theta):
        self.like_evals += 1
        theta = theta.clone().detach().requires_grad_(True)
        loglike = self.loglike(theta)

        if self.given_score:
            score = self.score(theta)
        else:
            loglike.backward()
            score = theta.grad
        if torch.isnan(score).any():
            raise ValueError("Score is NaN for theta = {}".format(theta))
        return loglike, score


    def langevin(self, min_like):
        idx = np.random.randint(self.nlive-1)
        #curr_value = self.live_points.get_values()[idx]
        curr_value = self.sample_prior(1).get_values()[0]
        loglike, score = self.get_score(curr_value)
        converged = False
        num_steps = 0
        accepted = 0
        # alpha = 1
        logalpha = torch.rand(1).item() * 5 - 6
        alpha = 10**logalpha
        while not converged:
            curr_value = curr_value + alpha*(score + torch.randn(curr_value.shape, device=self.device))
            curr_value = torch.clamp(curr_value, self._lower, self._upper)

            # if loglike > min_like:
            #     curr_value = curr_value + alpha * np.sqrt(2.) * torch.randn(curr_value.shape)
            #     accepted += 1
            # else:
            #     curr_value = curr_value + score #+ np.sqrt(2.) * torch.randn(curr_value.shape)

            loglike, score = self.get_score(curr_value)
            num_steps += 1
            converged = (loglike > min_like) #and (accepted >= self.n_repeats)
            if accepted >= self.n_repeats:
                accepted = 0

            if not converged:
                logalpha = torch.rand(1).item() * 5 - 5
                alpha = 10 ** logalpha

                #alpha = np.clip(alpha * 0.5, 1e-8, 1)

        #print(accepted, rejected)

        sample = NSPoints(self.nparams)
        sample.add_samples(values=curr_value.reshape(1, -1),
                           logL=loglike.reshape(1),
                           weights=torch.ones(1, device=self.device))
        return sample

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
        while newlike < min_like:
            newsample = self.langevin(min_like)
            newlike = newsample.get_logL()[0]

            self.n_tries += 1

        return newsample

if __name__ == "__main__":
    ndims = 100
    mvn1 = torch.distributions.MultivariateNormal(loc=2*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-1*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    def get_loglike(theta):
        lp = torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=-1, keepdim=False)
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

    ns = LangevinNest(
        nlive=25*len(params),
        loglike=get_loglike,
        params=params)

    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 / 10**len(params)))
    print('Number of evaluations', ns.get_like_evals())

    from getdist import plots
    samples = ns.convert_to_getdist()
    g = plots.get_subplot_plotter()
    g.triangle_plot([samples], filled=True)
    g.export('test.png')