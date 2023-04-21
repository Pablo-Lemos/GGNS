import torch
from torchNS.nested_sampling import NestedSampler
from torchNS.param import Param, NSPoints
from random import randint
from numpy import clip

# Default floating point type
dtype = torch.float32

class ReflectNest(NestedSampler):
    def __init__(self, loglike, params, nlive=50, tol=0.1, dt_ini=0.5, num_repeats=None, max_nsteps=1000000, verbose=True, score=None, device=None):
        super().__init__(loglike, params, nlive, tol, max_nsteps, verbose, device)

        self.acceptance_rate = 1.0
        self.n_tries = 0
        self.n_accepted = 0
        self.n_repeats = int(2 * self.nparams) if num_repeats is None else num_repeats

        self.dt = dt_ini
        self.n_in_steps = 0
        self.n_out_steps = 0

        self.given_score = score is not None
        if self.given_score:
            self.score = score

        self.n_tries_pure_ns = 0
        self.n_accepted_pure_ns = 0
        self.acc_rate_pure_ns = 1

        self.max_size = torch.tensor(.1)

        self._lower = torch.tensor([p.prior[0] for p in self.params], dtype=dtype, device=self.device)
        self._upper = torch.tensor([p.prior[1] for p in self.params], dtype=dtype, device=self.device)
        #assert p.prior_type == "uniform" for p in self.params, "Prior must be uniform for now"

    def random_walk(self, position, min_like, dt, num_steps):
        """
        Simulate the motion of a particle in a box with walls defined by the function p(X) = p0,
        where X is a three-vector (x, y, z), using PyTorch.

        Args:
            position (torch.Tensor): Initial position of the particle, shape (3,).
            velocity (torch.Tensor): Initial velocity of the particle, shape (3,).
            p_func (callable): Function that computes p(X) for a given three-vector X.
            p0 (float): Value of p0 for the walls of the box.
            dt (float): Time step for numerical integration.
            num_steps (int): Number of time steps to simulate.

        Returns:
            position_history (torch.Tensor): History of particle positions, shape (num_steps+1, 3).
        """
        reflected = False
        step = 0
        n_failed = 0

        while step < num_steps:
            if not reflected:
                velocity = torch.randn(self.nparams, device=self.device)

            position += velocity * dt
            p_x, grad_p_x = self.get_score(position)

            if p_x <= min_like:
                #if reflected:
                #    raise ValueError("Particle got stuck at the boundary")
                # Reflect velocity using the normal vector of the wall
                normal = grad_p_x / torch.norm(grad_p_x)
                velocity -= 2 * torch.dot(velocity, normal) * normal
                reflected = True
                n_failed += 1
                self.n_out_steps += 1
                if n_failed == 5:
                    return torch.zeros_like(position), - torch.inf

            else:
                reflected = False
                step += 1
                n_failed = 0
                self.n_in_steps += 1

                # Move position slightly inside the walls to avoid getting stuck at the boundary
                # position -= normal * (p_x - min_like)

        return position, p_x

    def random_walk_v2(self, position, min_like, dt, num_steps):
        """
        Simulate the motion of a particle in a box with walls defined by the function p(X) = p0,
        where X is a three-vector (x, y, z), using PyTorch.

        Args:
            position (torch.Tensor): Initial position of the particle, shape (3,).
            velocity (torch.Tensor): Initial velocity of the particle, shape (3,).
            p_func (callable): Function that computes p(X) for a given three-vector X.
            p0 (float): Value of p0 for the walls of the box.
            dt (float): Time step for numerical integration.
            num_steps (int): Number of time steps to simulate.

        Returns:
            position_history (torch.Tensor): History of particle positions, shape (num_steps+1, 3).
        """

        reflected = False
        for step in range(num_steps):
            if not reflected:
                velocity = torch.randn(self.nparams, device=self.device)

            position += velocity * dt
            #theta = position.clone().detach().requires_grad_(True)
            p_x, grad_p_x = self.get_score(position)

            if p_x <= min_like:
                if reflected:
                    raise ValueError("Particle got stuck at the boundary")
                # Reflect velocity using the normal vector of the wall
                normal = grad_p_x / torch.norm(grad_p_x)
                velocity -= 2 * torch.dot(velocity, normal) * normal
                reflected = True
                self.n_out_steps += 1
            else:
                reflected = False
                self.n_in_steps += 1

                # Move position slightly inside the walls to avoid getting stuck at the boundary
                # position -= normal * (p_x - min_like)

        return position, p_x

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

    def reflect_sampling(self, min_loglike):
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

        num_steps = self.n_repeats
        idx = randint(0, self.nlive - 2)
        x = self.live_points.get_values()[idx]

        accepted = False
        num_fails = 0
        while not accepted:
            new_x, new_loglike = self.random_walk(position=x, min_like=min_loglike, dt=0.1, num_steps=num_steps)
            accepted = new_loglike > min_loglike[0]

            acceptance = self.n_in_steps / (self.n_out_steps + self.n_in_steps)
            if acceptance > 0.5:
                self.dt = clip(1.1 * self.dt, 1e-5, 1.)
            elif acceptance < 0.2:
                self.dt = clip(0.9 * self.dt, 1e-5, 1.)

            if not accepted:
                num_fails += 1
                idx = randint(0, self.nlive - 2)
                x = self.live_points.get_values()[idx]

        assert new_loglike > min_loglike[0], "loglike = {}, min_loglike = {}".format(loglike, min_loglike)

        loglike1, score = self.get_score(new_x)
        assert loglike1 == new_loglike, "loglike1 = {}, loglike2 = {}".format(loglike1, new_loglike)

        sample = NSPoints(self.nparams)
        sample.add_samples(values=new_x.reshape(1, -1),
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
            #if self.n_accepted < 5*self.nlive:
            if self.acc_rate_pure_ns > 0.1:
                newsample = self.sample_prior(npoints=1)
                pure_ns = True
            else:
                newsample = self.reflect_sampling(min_like)
                pure_ns = False

            newlike = newsample.get_logL()[0]

            self.n_tries += 1
            if pure_ns: self.n_tries_pure_ns += 1

        self.n_accepted += 1
        if pure_ns:
            self.n_accepted_pure_ns += 1
            self.acc_rate_pure_ns = self.n_accepted_pure_ns / self.n_tries_pure_ns

        return newsample



if __name__ == "__main__":
    import time
    ndims = 10
    mvn1 = torch.distributions.MultivariateNormal(loc=2*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-1*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    true_samples = torch.cat([mvn1.sample((5000,)), mvn2.sample((5000,))], dim=0)
    #print(true_samples.shape)

    def get_loglike(theta):
        lp = torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=-1, keepdim=False) - torch.log(torch.tensor(2.0))
        return lp
        #return mvn1.log_prob(theta)

    params = []

    for i in range(ndims):
        params.append(
            Param(
                name=f'p{i}',
                prior_type='Uniform',
                prior=(-5, 5),
                label=f'p_{i}')
        )

    ns = ReflectNest(
        nlive=25*len(params),
        loglike=get_loglike,
        params=params)

    start_time = time.time()
    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 / 10**len(params)))
    print('Number of evaluations', ns.get_like_evals())
    print('Time taken', time.time() - start_time)

    from getdist import plots, MCSamples
    samples = ns.convert_to_getdist()
    true_samples = MCSamples(samples=true_samples.numpy(), names=[f'p{i}' for i in range(ndims)])
    g = plots.get_subplot_plotter()
    g.triangle_plot([true_samples, samples], filled=True, legend_labels=['True', 'GDNest'])
    g.export('test_reflect.png')