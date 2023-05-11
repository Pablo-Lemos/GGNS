import torch
from torchNS.nested_sampling import NestedSampler
from torchNS.param import Param, NSPoints
from random import randint

# Default floating point type
dtype = torch.float32


class Polychord(NestedSampler):
    def __init__(self, loglike, params, nlive=50, tol=0.1, max_nsteps=1000000, verbose=True, score=None, clustering=False, device=None):
        super().__init__(loglike, params, nlive, tol, max_nsteps, verbose=verbose, clustering=clustering, device=device)

        self.acceptance_rate = 1.0
        self.n_tries = 0
        self.n_accepted = 0
        self.n_repeats = 2 * int(self.nparams)

        self.given_score = score is not None
        if self.given_score:
            self.score = score

        self._lower = torch.tensor([p.prior[0] for p in self.params], dtype=dtype, device=self.device)
        self._upper = torch.tensor([p.prior[1] for p in self.params], dtype=dtype, device=self.device)
        #assert p.prior_type == "uniform" for p in self.params, "Prior must be uniform for now"

    def slice_sampling(self, log_slice_height, initial_x, step_size=1):
        """
        Slice sampling algorithm for PyTorch.

        Arguments:
        log_prob_func -- A function that takes a PyTorch tensor and returns its log probability.
        initial_x -- A PyTorch tensor representing the initial value of x.
        num_samples -- The number of samples to generate.
        step_size -- The step size used in the algorithm.

        Returns:
        samples -- A PyTorch tensor of shape (num_samples,) representing the generated samples.
        """

        x = initial_x.clone()
        d = torch.randn(self.nparams, device=self.device)
        d = d / torch.linalg.norm(d)
        # Choose a random slice height
        # log_slice_height = log_prob_func(x) - torch.rand(1).item()

        temp_random = torch.rand(1, device=self.device)
        w = 1
        # Set the initial interval
        L = x - temp_random * w * d
        R = x + (1 - temp_random) * w * d

        # Shrink the interval until it brackets the slice

        # For left and right
        self.like_evals += 2

        # TODO this only works for uniform priors where all parameters have the same limits
        while (self.loglike(L) > log_slice_height) and (torch.min(L) > torch.min(self._lower) and (torch.max(L) < torch.max(self._upper))):
            self.like_evals += 1
            L = L - d * w * step_size
            #L = torch.clamp(L, self._lower, self._upper)
        while (self.loglike(R) > log_slice_height) and (torch.max(R) < torch.max(self._upper) and (torch.min(R) > torch.min(self._lower))):
            self.like_evals += 1
            R = R + d * w * step_size
            #R = torch.clamp(R, self._lower, self._upper)

        # Sample a new x within the interval
        while True:
            #print(L, R)
            x0Ld = torch.linalg.norm((L - x))#torch.sum((x - L) ** 2, dim=0)**0.5
            x0Rd = torch.linalg.norm((R - x))#torch.sum((x - R) ** 2, dim=0)**0.5

            new_x = x + (torch.rand(1, device=self.device) * (x0Rd + x0Ld) - x0Ld) * d
            #new_x = torch.clamp(new_x, self._lower, self._upper)
            new_log_prob, _ = self.get_score(new_x)

            if new_log_prob > log_slice_height:
                x = new_x
                break
            else:
                if (new_x - x).dot(d) < 0:
                    L = new_x.clone()
                else:
                    R = new_x.clone()

        return x, new_log_prob


    def grad_desc(self, min_like):
        cluster_volumes = torch.exp(self.summaries.get_logXp())
        curr_value = self.live_points.get_random_sample(cluster_volumes).get_values()[0]

        for i in range(self.n_repeats):
            new_value, new_loglike = self.slice_sampling(min_like, curr_value)

        sample = NSPoints(self.nparams)
        sample.add_samples(values=new_value.reshape(1, -1),
                           logL=new_loglike.reshape(1),
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
            newsample = self.grad_desc(min_like)
            newlike = newsample.get_logL()[0]
            self.n_tries += 1

        self.n_accepted += 1

        return newsample


if __name__ == "__main__":
    import time
    ndims = 32
    mvn = torch.distributions.MultivariateNormal(loc=torch.zeros(ndims),
                                             scale_tril=torch.diag(
                                                 torch.ones(ndims)))

    mvn1 = torch.distributions.MultivariateNormal(loc=0*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-1*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    #true_samples = torch.cat([mvn1.sample((5000,)), mvn2.sample((5000,))], dim=0)
    #print(true_samples.shape)
    true_samples = mvn.sample((20000,))

    def get_loglike(theta):
        #lp = torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=-1, keepdim=False) - torch.log(torch.tensor(2.0))
        #return lp
        return mvn1.log_prob(theta)

    params = []

    for i in range(ndims):
        params.append(
            Param(
                name=f'p{i}',
                prior_type='Uniform',
                prior=(-10, 10),
                label=f'p_{i}')
        )

    ns = Polychord(
        nlive=25*len(params),
        loglike=mvn.log_prob,#get_loglike,
        params=params,
        clustering=False)

    start_time = time.time()
    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 / 20**len(params)))
    print('Number of evaluations', ns.get_like_evals())
    print('Time taken', time.time() - start_time)

    from getdist import plots, MCSamples
    samples = ns.convert_to_getdist()
    true_samples = MCSamples(samples=true_samples.numpy(), names=[f'p{i}' for i in range(ndims)])
    g = plots.get_subplot_plotter()
    g.triangle_plot([true_samples, samples], [f'p{i}' for i in range(5)], filled=True, legend_labels=['True', 'GDNest'])
    g.export('test_polychord.png')
