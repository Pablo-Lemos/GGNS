import torch
from torchNS.nested_sampling import NestedSampler
from torchNS.param import Param, NSPoints
from random import randint
from numpy import clip, pi

# Default floating point type
dtype = torch.float32

class GaliNest(NestedSampler):
    def __init__(self, loglike, params, nlive=50, tol=0.1, dt_ini=0.1, max_nsteps=1000000, clustering=False, verbose=True, score=None, device=None):
        super().__init__(loglike, params, nlive, tol, max_nsteps, clustering, verbose, device)

        self.acceptance_rate = 1.0
        self.n_tries = 0
        self.n_repeats = int(2 * self.nparams)

        self.dt = dt_ini
        self.n_in_steps = 0
        self.n_out_steps = 0

        self.given_score = score is not None
        if self.given_score:
            self.score = score

        self.n_tries_pure_ns = 0
        self.n_accepted_pure_ns = 0
        self.acc_rate_pure_ns = 1

    #@torch.compile
    def simulate_particle_in_box(self, position, velocity, min_like, dt, max_reflections):
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
        n_reflections = 0
        last_reflections = torch.zeros_like(position, dtype=dtype, device=self.device)
        new_reflections = torch.zeros_like(position, dtype=dtype, device=self.device)
        positions = []
        loglikes = []
        last_in = torch.ones(position.shape[0], dtype=torch.bool, device=self.device)

        while n_reflections < max_reflections:
            position += velocity * dt
            p_x, grad_p_x = self.get_score(position)

            outside = (p_x <= min_like) + ~self.is_in_prior(position)
            normal = grad_p_x / torch.norm(grad_p_x, dim=-1)
            v_dot_normal = torch.einsum('ai, ai -> a', velocity, normal)
            dealigned = v_dot_normal < 0
            # Only reflect when the particle is outside the wall, and its velocity its pointed away from the normal
            reflected = outside * dealigned
            delta_velocity = 2 * v_dot_normal.reshape(-1, 1) * normal
            velocity[reflected, :] -= delta_velocity[reflected, :]
            n_reflections += reflected.sum()

            last_reflections[last_in * reflected, :] = new_reflections[last_in * reflected, :]
            new_reflections[last_in * reflected, :] = position[last_in * reflected, :]

            last_in = ~outside

            # self.n_out_steps += reflected.sum()
            # self.n_in_steps += (~reflected).sum()

            # if (n_reflections > max_reflections/2):
            #     positions.append(position[~outside, :])
            #     loglikes.append(p_x[~outside])

        # positions = torch.cat(positions, dim=0)
        # loglikes = torch.cat(loglikes, dim=0)
        # if positions.shape[0] == 0:
        #     return position, -1e30 * torch.ones_like(p_x)
        # else:
        #     # Select a position from the history of positions
        #     idx = randint(0, positions.shape[0] - 1)
        #     position = positions[idx, :]
        #     p_x = loglikes[idx]
        #
        #     return position.unsqueeze(0), p_x
        if torch.any(last_reflections) == 0:
            return position, -1e30 * torch.ones_like(p_x)
        else:
            segments = new_reflections - last_reflections

            # Generate random values between 0 and 1
            random_values = torch.rand((position.shape[0], 1))

            # Multiply the line segments by the random values
            # to get the points along the segments
            pos_out = last_reflections + random_values * segments
            logl_out, _ = self.get_score(pos_out)
            mask2 = logl_out < min_like

            logl_out[mask2] = -1e30

            return pos_out, logl_out#, out_frac

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
        initial_point = self.live_points.get_random_sample(cluster_volumes)
        x = initial_point.get_values()

        accepted = False
        num_fails = 0
        while not accepted:
            r = torch.randn_like(x, dtype=dtype, device=self.device)
            velocity = r / torch.linalg.norm(r, dim=-1, keepdim=True)
            new_x, new_loglike = self.simulate_particle_in_box(position=x, velocity=velocity, min_like=min_loglike,
                                                               dt=self.dt, max_reflections=5)

            #in_prior = (torch.min(new_x - self._lower, dim=-1)[0] >= torch.zeros(new_x.shape[0])) * (torch.max(new_x - self._upper, dim=-1)[0] <= torch.zeros(new_x.shape[0]))
            in_prior = self.is_in_prior(new_x)
            accepted = (new_loglike > min_loglike) * in_prior

            if not accepted:
                num_fails += 1
                initial_point = self.live_points.get_random_sample(cluster_volumes)
                x = initial_point.get_values()

        assert self.is_in_prior(new_x), "new_x = {}, lower = {}, upper = {}".format(new_x, self._lower, self._upper)
        assert new_loglike > min_loglike[0], "loglike = {}, min_loglike = {}".format(loglike, min_loglike)
        assert self.loglike(new_x) == new_loglike, "loglike = {}, new_loglike = {}".format(self.loglike(new_x), new_loglike)

        sample = NSPoints(self.nparams)
        sample.add_samples(values=new_x.reshape(1, -1),
                           logL=new_loglike.reshape(1),
                           logweights=torch.ones(1, device=self.device),
                           labels=initial_point.get_labels())
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
            if self.acc_rate_pure_ns > 1.1:
                newsample = self.sample_prior(npoints=1)
                pure_ns = True
            else:
                newsample = self.reflect_sampling(min_like)
                pure_ns = False

            newlike = newsample.get_logL()[0]

            self.n_tries += 1
            if pure_ns: self.n_tries_pure_ns += 1

        if pure_ns:
            self.n_accepted_pure_ns += 1
            self.acc_rate_pure_ns = self.n_accepted_pure_ns / self.n_tries_pure_ns

        return newsample


if __name__ == "__main__":
    ndims = 5
    mvn1 = torch.distributions.MultivariateNormal(loc=0*torch.ones(ndims, dtype=dtype),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims, dtype=dtype)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-1*torch.ones(ndims, dtype=dtype),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims, dtype=dtype)))

    #true_samples = torch.cat([mvn1.sample((5000,)), mvn2.sample((5000,))], dim=0)
    true_samples = mvn1.sample((5000,))

    def get_loglike(theta):
        lp = mvn1.log_prob(theta)
        #mask = (torch.min(theta, dim=-1)[0] >= -5) * (torch.max(theta, dim=-1)[0] <= 5)
        return lp #- 1e30 * (1 - mask.float())

    params = []

    for i in range(ndims):
        params.append(
            Param(
                name=f'p{i}',
                prior_type='Uniform',
                prior=(-5, 5),
                label=f'p_{i}')
        )

    ns = GaliNest(
        nlive=25*len(params),
        loglike=get_loglike,
        params=params,
        clustering=False,
        tol=1e-2
    )

    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 / 10**len(params)))
    print('Number of evaluations', ns.get_like_evals())

    from getdist import plots, MCSamples
    samples = ns.convert_to_getdist()
    true_samples = MCSamples(samples=true_samples.numpy(), names=[f'p{i}' for i in range(ndims)])
    g = plots.get_subplot_plotter()
    g.triangle_plot([true_samples, samples], [f'p{i}' for i in range(5)], filled=True, legend_labels=['True', 'GDNest'])
    g.export('./plots/test_galilean.png')
