import torch
from torchNS.dynamic import DynamicNestedSampler
from torchNS.param import Param, NSPoints
from random import randint
from numpy import clip, pi

# Default floating point type
dtype = torch.float32

class DyGaliNest(DynamicNestedSampler):
    def __init__(self, loglike, params, nlive=50, tol=0.1, dt_ini=0.5, max_nsteps=1000000, clustering=False, verbose=True, score=None, device=None):
        super().__init__(loglike=loglike,
                         params=params,
                         nlive=nlive,
                         tol=tol,
                         max_nsteps=max_nsteps,
                         clustering=clustering,
                         verbose=verbose,
                         device=device)
        # loglike, params, nlive=50, tol=0.1, max_nsteps=1000000, clustering=False, verbose=True, device=None

        self.acceptance_rate = 1.0
        self.n_tries = 0
        self.n_repeats = int(5 * self.nparams)

        self.dt = dt_ini
        self.n_in_steps = 0
        self.n_out_steps = 0

        self.given_score = score is not None
        if self.given_score:
            self.score = score

        self.max_size = torch.tensor(.1)

        self.n_tries_pure_ns = 0
        self.n_accepted_pure_ns = 0
        self.acc_rate_pure_ns = 1

        self._lower = torch.tensor([p.prior[0] for p in self.params], dtype=dtype, device=self.device)
        self._upper = torch.tensor([p.prior[1] for p in self.params], dtype=dtype, device=self.device)
        #assert p.prior_type == "uniform" for p in self.params, "Prior must be uniform for now"

    #@torch.compile
    def simulate_particle_in_box(self, position, velocity, min_like, dt, num_steps):
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
        assert(len(position.shape) == 2), "Position must be a 2D tensor"
        for step in range(num_steps):
            # reflected = False
            position += velocity * dt
            # Slightly perturb the position to decorrelate the samples
            position *= (1 + 1e-2 * torch.randn_like(position))
            p_x, grad_p_x = self.get_score(position)

            reflected = p_x <= min_like
            normal = grad_p_x / torch.norm(grad_p_x, dim=1, keepdim=True)
            #delta_velocity = 2 * torch.tensordot(velocity, normal, dims=([1], [1])) * normal
            delta_velocity = 2 * torch.einsum('ai, ai -> a', velocity, normal).reshape(-1, 1) * normal
            velocity[reflected, :] -= delta_velocity[reflected, :]
            self.n_out_steps += reflected.sum()
            self.n_in_steps += (~reflected).sum()

            # if p_x <= min_like:
            #     # if reflected:
            #     #     raise ValueError("Particle got stuck at the boundary")
            #     # Reflect velocity using the normal vector of the wall
            #     normal = grad_p_x / torch.norm(grad_p_x)
            #     velocity -= 2 * torch.dot(velocity, normal) * normal
            #     # reflected = True
            #     self.n_out_steps += 1
            # else:
            #     # reflected = False
            #     self.n_in_steps += 1

        return position, p_x

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
        cluster_volumes = torch.exp(self.summaries.get_logXp())
        #x = self.live_points.get_random_sample(cluster_volumes, n_samples=self.nlive_ini//2).get_values()
        #alpha = 1
        point = self.live_points.get_random_sample(cluster_volumes, n_samples=self.nlive_ini//2)
        x = point.get_values()
        labels = point.get_labels()
        dt = 0.1

        log_gamma = torch.lgamma(torch.tensor(self.nparams / 2 + 1))
        # Use the torch.exp function to compute the exponential of the log
        gamma = torch.exp(log_gamma)
        # Use the formula for the radius in terms of volume and dimension

        alpha = self.nparams * (cluster_volumes[labels] * gamma / pi ** (self.nparams / 2)) ** (1 / self.nparams)
        
        alpha = 1.
        accepted = False
        num_fails = 0
        active = torch.ones(x.shape[0], dtype=torch.bool)
        new_x = torch.zeros_like(x)
        new_loglike = torch.zeros(x.shape[0], dtype=torch.float32)
        while not accepted:
            r = torch.randn_like(x)
            #velocity = alpha.reshape(-1, 1) * r / torch.norm(r, dim=-1, keepdim=True)
            velocity = alpha * r
            #print(active.shape, x[active].shape, velocity[active].shape)
            new_x_active, new_loglike_active = self.simulate_particle_in_box(position=x[active], velocity=velocity[active], min_like=min_loglike, dt=dt, num_steps=self.n_repeats)
            new_x[active] = new_x_active
            new_loglike[active] = new_loglike_active
            active = new_loglike < min_loglike
            #print(torch.sum(active))
            accepted = torch.sum(active) == 0
            #print("Accepted: ", torch.sum(new_loglike > min_loglike).item(), " / ", len(new_loglike))

            if not accepted:
                point = self.live_points.get_random_sample(cluster_volumes, n_samples=self.nlive_ini // 2)
                x = point.get_values()
                labels = point.get_labels()
                alpha = self.nparams * (cluster_volumes[labels] * gamma / pi ** (self.nparams / 2)) ** (1 / self.nparams)
                alpha = 1.

        sample = NSPoints(self.nparams)
        sample.add_samples(values=new_x,
                           logL=new_loglike,
                           weights=torch.ones(new_loglike.shape[0], device=self.device))
        return sample


    def find_new_sample_batch(self, min_like, n_points):
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
        newsamples = self.reflect_sampling(min_like)

        return newsamples

    def move_one_step(self):
        ''' Find highest log likes, get rid of those point, and sample a new ones '''
        if self.acc_rate_pure_ns > 0.1:
            sample = self.kill_point()
            min_like = sample.get_logL()
            newlike = -torch.inf
            while newlike < min_like:
                newsample = self.sample_prior(npoints=1)
                newlike = newsample.get_logL()[0]
                self.n_tries_pure_ns += 1

            if self.clustering:
                # Find closest point to new sample
                values = self.live_points.get_values()
                dist = torch.sum((values - newsample.get_values())**2, dim=1)
                idx = torch.argmin(dist)

                # Assign its label to the new point
                newsample.set_labels(self.live_points.get_labels()[idx].reshape(1))

            self.n_accepted += 1
            self.n_accepted_pure_ns += 1
            self.acc_rate_pure_ns = self.n_accepted_pure_ns / self.n_tries_pure_ns
            self.live_points.add_nspoint(newsample)
        else:
            n_points = self.nlive_ini//2
            for _ in range(n_points):
                sample = self.kill_point()

            self.add_point_batch(min_logL=sample.get_logL(), n_points=n_points)


if __name__ == "__main__":
    ndims = 20
    mvn1 = torch.distributions.MultivariateNormal(loc=2*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-1*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    true_samples = torch.cat([mvn1.sample((5000,)), mvn2.sample((5000,))], dim=0)

    def get_loglike(theta):
        lp = torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=0, keepdim=False) - torch.log(torch.tensor(2.0))
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

    ns = DyGaliNest(
        nlive=25*len(params),
        loglike=get_loglike,
        params=params,
        verbose=True,
        clustering=True,
        tol=1e-1
    )

    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 / 10**len(params)))
    print('Number of evaluations', ns.get_like_evals())

    # from getdist import plots, MCSamples
    # samples = ns.convert_to_getdist()
    # true_samples = MCSamples(samples=true_samples.numpy(), names=[f'p{i}' for i in range(ndims)])
    # g = plots.get_subplot_plotter()
    # g.triangle_plot([true_samples, samples], filled=True, legend_labels=['True', 'GDNest'])
    # g.export('test_dygalilean.png')
