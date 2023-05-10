import torch
from torchNS.nested_sampling import NestedSampler
from torchNS.param import Param, NSPoints
from random import randint
from numpy import clip

# Default floating point type
dtype = torch.float64

class DyGaliNest(NestedSampler):
    def __init__(self, loglike, params, nlive=50, tol=0.1, dt_ini=0.1, max_nsteps=1000000, clustering=False, verbose=True, score=None, device=None):
        super().__init__(loglike, params, nlive, tol, max_nsteps, clustering, verbose, device)

        self.acceptance_rate = 1.0
        self.n_tries = 0
        self.n_accepted = 0
        self.n_repeats = 20 #int(2 * self.nparams)

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

        num_reflections = 0
        max_reflections = 4 #14
        min_reflections = 1 #10
        x = position.clone()
        positions = torch.tensor([], dtype=dtype, device=self.device)
        loglikes = torch.tensor([], dtype=dtype, device=self.device)
        while num_reflections < max_reflections:
        #for step in range(num_steps):
            reflected = False
            x += velocity * dt
            #theta = position.clone().detach().requires_grad_(True)
            p_x, grad_p_x = self.get_score(x)

            if p_x <= min_like:
                if reflected:
                    raise ValueError("Particle got stuck at the boundary")
                # Reflect velocity using the normal vector of the wall
                normal = grad_p_x / torch.norm(grad_p_x)
                velocity -= 2 * torch.dot(velocity, normal) * normal
                num_reflections += 1
                reflected = True
                self.n_out_steps += 1
            else:
                reflected = False
                self.n_in_steps += 1
                velocity *= (1 + 0.01*torch.rand_like(velocity))
                # Generate a new random velocity and adjust its magnitude

                if num_reflections > min_reflections:
                    positions = torch.cat((positions, x.unsqueeze(0)))
                    loglikes = torch.cat((loglikes, p_x.unsqueeze(0)))
                #velocity *= (1 + torch.randn_like(velocity) * 0.1)

                # Move position slightly inside the walls to avoid getting stuck at the boundary
                # position -= normal * (p_x - min_like)

        if len(positions) > 0:
            idx = torch.randint(0, positions.shape[0], (1,))[0]
            x = positions[idx]
            p_x = loglikes[idx]
        else:
            x = torch.zeros(self.nparams, dtype=dtype, device=self.device)
            p_x = torch.tensor(-1e300, dtype=dtype, device=self.device)

        return x, p_x, len(positions)/(max_reflections-min_reflections)

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
        # x = self.live_points.get_random_sample(cluster_volumes).get_values()[0]
        num_steps = self.n_repeats
        alpha = 1
        dt = 0.5

        point = self.live_points.get_random_sample(cluster_volumes)
        x = point.get_values()[0]
        #x = self.live_points.get_values()[self.idx]
        # labels = point.get_labels()
        # x2 = self.live_points.get_random_sample(cluster_volumes).get_values()[0]
        # while torch.allclose(x, x2):
        #      x2 = self.live_points.get_random_sample(cluster_volumes).get_values()[0]
        #

        A = torch.cov(self.live_points.get_values().T)
        L = torch.linalg.cholesky(A)

        accepted = False
        num_fails = 0
        while not accepted:
            r = torch.randn_like(x)
            #r /= torch.linalg.norm(r, dim=-1, keepdim=True)
            #velocity = r * (x - x2)
            #velocity = r @ self.L
            #velocity = r * torch.min(torch.diag(L))
            #velocity = r * torch.diag(L)
            velocity = r * torch.std(self.live_points.get_values().T)
            #velocity = r * torch.std(self.live_points.get_values().T)
            #velocity = r * torch.diag(A).sqrt()
            #velocity = alpha * torch.randn(self.nparams, device=self.device)
            new_x, new_loglike, num_pts = self.simulate_particle_in_box(position=x, velocity=velocity, min_like=min_loglike, dt=dt, num_steps=num_steps)

            #acceptance = self.n_in_steps / (self.n_out_steps + self.n_in_steps)
            #print(new_x, new_loglike, num_reflections)
            #print(self.loglike(x), self.loglike(new_x), min_loglike)
            #if acceptance > 0.5:
            # if num_pts < 10:
            #     self.dt = clip(0.9 * self.dt, 1e-5, 10.)
            #     #print(f"Decreasing dt to {self.dt}")
            # #elif acceptance < 0.2:
            # elif num_pts > 50:
            #     self.dt = clip(1.1 * self.dt, 1e-5, 10.)
            #     #print(f"Increasing dt to {self.dt}")
            # else:
            #     accepted = new_loglike > min_loglike[0]

            accepted = new_loglike > min_loglike[0]

            if not accepted:
                num_fails += 1
                #x = self.live_points.get_random_sample(self.cluster_volumes).get_values()[0]
                point = self.live_points.get_random_sample(cluster_volumes)
                #x = self.live_points.get_values()[self.idx]
                #x = point.get_values()[0]

        assert new_loglike > min_loglike[0], "loglike = {}, min_loglike = {}".format(loglike, min_loglike)
        assert self.loglike(new_x) == new_loglike, "loglike = {}, min_loglike = {}".format(loglike, min_loglike)

        #print(new_loglike, min_loglike)
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
            if self.acc_rate_pure_ns > 0.1:
                newsample = self.sample_prior(npoints=1)
                pure_ns = True
            else:
                A = torch.cov(self.live_points.get_values().T)
                self.L = torch.linalg.cholesky(A)
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

    # def move_one_step(self):
    #     for _ in range(self.nlive_ini//2):
    #         sample = self.kill_point()
    #         #print("killed point", sample.get_logL())
    #     logL = sample.get_logL()
    #     self.idx = 0
    #     A = torch.cov(self.live_points.get_values().T)
    #     self.L = torch.linalg.cholesky(A)
    #     for i in range(self.nlive_ini//2):
    #         #print(self.live_points.values.shape)
    #         self.add_point(logL)
    #         self.idx += 1


if __name__ == "__main__":
    ndims = 128
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
                prior=(-5, 5),
                label=f'p_{i}')
        )

    ns = DyGaliNest(
        nlive=25*len(params),
        loglike=mvn.log_prob,#get_loglike,
        params=params,
        clustering=False)
    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 /10**len(params)))
    print('Number of evaluations', ns.get_like_evals())

    from getdist import plots, MCSamples
    samples = ns.convert_to_getdist()
    true_samples = MCSamples(samples=true_samples.numpy(), names=[f'p{i}' for i in range(ndims)])
    g = plots.get_subplot_plotter()
    g.triangle_plot([true_samples, samples], [f'p{i}' for i in range(5)], filled=True, legend_labels=['True', 'GDNest'])
    g.export('test_dygalilean.png')
