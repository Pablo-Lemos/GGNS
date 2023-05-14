import torch
from torchNS.dynamic import DynamicNestedSampler
from torchNS.param import Param, NSPoints
from random import randint
from numpy import clip, pi

# Default floating point type
dtype = torch.float64

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
        self.n_repeats = int(self.nparams)

        self.dt = dt_ini
        #self.n_in_steps = 0
        #self.n_out_steps = 0

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

    # def simulate_particle_in_box(self, position, velocity, min_like, dt, num_steps):
    #     """
    #     Simulate the motion of a particle in a box with walls defined by the function p(X) = p0,
    #     where X is a three-vector (x, y, z), using PyTorch.
    #
    #     Args:
    #         position (torch.Tensor): Initial position of the particle, shape (3,).
    #         velocity (torch.Tensor): Initial velocity of the particle, shape (3,).
    #         p_func (callable): Function that computes p(X) for a given three-vector X.
    #         p0 (float): Value of p0 for the walls of the box.
    #         dt (float): Time step for numerical integration.
    #         num_steps (int): Number of time steps to simulate.
    #
    #     Returns:
    #         position_history (torch.Tensor): History of particle positions, shape (num_steps+1, 3).
    #     """
    #     assert(len(position.shape) == 2), "Position must be a 2D tensor"
    #     n_out_steps = 0
    #     n_in_steps = 0
    #     x = position.clone()
    #     for step in range(num_steps):
    #         # reflected = False
    #         x += velocity * dt #* (1 + 0.1 * torch.randn(1, dtype=dtype, device=self.device))
    #         # Slightly perturb the position to decorrelate the samples
    #         #position *= (1 + 1e-2 * torch.randn_like(position))
    #         p_x, grad_p_x = self.get_score(x)
    #
    #         reflected = p_x <= min_like
    #         normal = grad_p_x / torch.norm(grad_p_x, dim=1, keepdim=True)
    #         #delta_velocity = 2 * torch.tensordot(velocity, normal, dims=([1], [1])) * normal
    #         normal = normal.to(dtype)
    #         delta_velocity = 2 * torch.einsum('ai, ai -> a', velocity, normal).reshape(-1, 1) * normal
    #         velocity[reflected, :] -= delta_velocity[reflected, :] #* (1 + 1e-2 * torch.randn_like(velocity[reflected]))
    #
    #         r = torch.randn_like(velocity[~reflected], dtype=dtype, device=self.device)
    #         r /= torch.linalg.norm(r, dim=-1, keepdim=True)
    #         velocity[~reflected] = velocity[~reflected] * (1 + 1e-2 * r)
    #         n_out_steps += reflected.sum()
    #         n_in_steps += (~reflected).sum()
    #
    #     out_frac = n_out_steps / (n_out_steps + n_in_steps)
    #
    #     return x, p_x, out_frac



    #@torch.compile
    def simulate_particle_in_box(self, position, velocity, min_like, dt, max_steps):
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
        n_out_steps = 0
        n_in_steps = 0
        min_reflections = 0
        max_reflections = 5
        num_reflections = torch.zeros(position.shape[0], dtype=torch.int64, device=self.device)
        num_inside_steps = torch.zeros(position.shape[0], dtype=torch.int64, device=self.device)
        #for step in range(num_steps):
        pos_ls = []
        logl_ls = []
        mask = []
        step = 0
        x = position.clone()
        while (torch.min(num_reflections) < max_reflections) and (step < max_steps):
            # reflected = False
            x += velocity * dt
            in_prior = (torch.min(x - self._lower, dim=-1)[0] >= torch.zeros(x.shape[0])) * (
                        torch.max(x - self._upper, dim=-1)[0] <= torch.zeros(x.shape[0]))
            # Slightly perturb the position to decorrelate the samples
            #position *= (1 + 1e-2 * torch.randn_like(position))
            p_x, grad_p_x = self.get_score(x)
            step += 1

            reflected = p_x <= min_like
            normal = grad_p_x / torch.norm(grad_p_x, dim=1, keepdim=True)
            #delta_velocity = 2 * torch.tensordot(velocity, normal, dims=([1], [1])) * normal
            normal = normal.to(dtype)
            delta_velocity = 2 * torch.einsum('ai, ai -> a', velocity, normal).reshape(-1, 1) * normal
            velocity[reflected, :] -= delta_velocity[reflected, :] #* (1 + 1e-2 * torch.randn_like(velocity[reflected]))

            num_reflections += reflected
            num_inside_steps += ~reflected
            if torch.min(num_reflections) > min_reflections:
                pos_ls.append(x.clone())
                logl_ls.append(p_x.clone())
                #mask.append(~reflected)
                mask.append(~reflected * in_prior)

            # v_norm = torch.linalg.norm(velocity, dim=-1, keepdim=True)
            r = torch.randn_like(velocity[~reflected], dtype=dtype, device=self.device)
            r /= torch.linalg.norm(r, dim=-1, keepdim=True)
            velocity[~reflected] = velocity[~reflected] * (1 + 5e-2 * r)
            # velocity[~reflected] = velocity[~reflected] * (1 + 1e-2 * torch.randn_like(velocity[~reflected]))
            # velocity[~reflected] = velocity[~reflected] / torch.linalg.norm(velocity[~reflected], dim=-1, keepdim=True) * v_norm[~reflected]
            n_out_steps += reflected.sum()
            n_in_steps += (~reflected).sum()
            #n_out_steps += (reflected + ~in_prior).sum()
            #n_in_steps += (~reflected * in_prior).sum()

        out_frac = n_out_steps / (n_out_steps + n_in_steps)

        if len(pos_ls) == 0:
            pos_out = torch.zeros_like(x)
            logl_out = torch.tensor(-1e30, dtype=torch.float64) * torch.ones(position.shape[0], dtype=dtype, device=self.device)

        else:
            positions = torch.stack(pos_ls, dim=0)
            loglikes = torch.stack(logl_ls, dim=0)
            masks = torch.stack(mask, dim=0)
            #idx = torch.randint(0, positions.shape[0] - 2, (position.shape[0],))
            pos_out = torch.zeros(position.shape, dtype=dtype, device=self.device)
            logl_out = torch.zeros(position.shape[0], dtype=dtype, device=self.device)

        #print(positions.shape)
        # if len(positions) == 0:
        #     pos_out = torch.zeros_like(x)
        #     logl_out = torch.tensor(1e-100, dtype=torch.float64)
        # else:
            for i in range(positions.shape[1]):
                pos = positions[:, i, :]
                ll = loglikes[:, i]

                if torch.sum(masks[:, i]) == 0:
                    pos_out[i, :] = 0.
                    logl_out[i] = -1e30
                else:
                    pos = pos[masks[:,i]]
                    ll = ll[masks[:,i]]
                    idx = torch.randint(0, pos.shape[0], (1,))
                    pos_out[i, :] = pos[idx, :].clone()
                    logl_out[i] = ll[idx].clone()

        #print(out_frac)
        return pos_out, logl_out, out_frac #torch.min(num_inside_steps/num_reflections)


    def reflect_sampling(self, min_loglike, labels=None):
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
        if self.n_clusters > 1:
            #cluster_volumes = torch.exp(self.summaries.get_logXp())
            #point = self.live_points.get_random_sample(labels, n_samples=torch.sum(labels).item())
            point = self.live_points.get_samples_from_labels(labels)
            x_ini = point.get_values()
            #labels = point.get_labels()
        else:
            x_ini = self.live_points.get_values()[:self.nlive_ini//2]
        #dt = 0.1

        alpha = 1.
        #num_fails = 0

        # A = torch.cov(self.live_points.get_values().T)
        # L = torch.linalg.cholesky(A)


        active = torch.ones(x_ini.shape[0], dtype=torch.bool)
        new_x = torch.zeros_like(x_ini, dtype=torch.float64)
        new_loglike = torch.zeros(x_ini.shape[0], dtype=torch.float64)

        accepted = False
        #velocity = r @ L.T
        #velocity = r * torch.diag(L)
        while not accepted:
            # assert torch.min(
            #     self.loglike(x_ini)) >= min_loglike, f"min_loglike = {min_loglike}, x_loglike = {self.loglike(x_ini)}"

            r = torch.randn_like(x_ini, dtype=dtype, device=self.device)
            r /= torch.linalg.norm(r, dim=-1, keepdim=True)
            velocity = alpha * r
            #r = torch.randn_like(x)
            #r /= torch.norm(r, dim=-1, keepdim=True)
            #velocity = alpha.reshape(-1, 1) * r / torch.norm(r, dim=-1, keepdim=True)
            #velocity = alpha * r
            #print(active.shape, x[active].shape, velocity[active].shape)

            assert (torch.min(x_ini - self._lower).item() >= 0) and (torch.max(x_ini - self._upper).item() <= 0)

            new_x_active, new_loglike_active, out_frac = self.simulate_particle_in_box(position=x_ini[active],
                                                                                       velocity=velocity[active],
                                                                                       min_like=min_loglike,
                                                                                       dt=self.dt,
                                                                                       max_steps=100*self.nparams)
            new_x[active] = new_x_active.to(dtype)
            new_loglike[active] = new_loglike_active.to(dtype)
            #print("Accepted: ", torch.sum(new_loglike > min_loglike).item(), " / ", len(new_loglike))

            if out_frac > 0.2:
                #self.dt *= 0.9
                self.dt = clip(self.dt * 0.9, 1e-5, 10)
                #active = torch.ones(x_ini.shape[0], dtype=torch.bool)
                if self.verbose: print("Decreasing dt to ", self.dt)
            elif out_frac < 0.05:
                #self.dt *= 1.1
                self.dt = clip(self.dt * 1.1, 1e-5, 10)
                #active = torch.ones(x_ini.shape[0], dtype=torch.bool)
                if self.verbose: print("Increasing dt to ", self.dt)
            else:
                in_prior = (torch.min(new_x - self._lower, dim=-1)[0] >= torch.zeros(new_x.shape[0])) * (torch.max(new_x - self._upper, dim=-1)[0] <= torch.zeros(new_x.shape[0]))
                active = (new_loglike < min_loglike) + (~in_prior)
                #print("Loglike: ", torch.sum(new_loglike < min_loglike).item(), " / ", len(new_loglike))
                #print(f"Active: {torch.sum(active).item()} / {len(active)}")
                #active = ~in_prior

            accepted = torch.sum(active) == 0
            #if not accepted:
                #if self.verbose: print(f"Active: {torch.sum(active).item()} / {len(active)}")
                # point = self.live_points.get_random_sample(cluster_volumes, n_samples=torch.sum(active).item())
                # x_ini[active] = point.get_values()

        assert torch.min(new_loglike) >= min_loglike, f"min_loglike = {min_loglike}, new_loglike = {new_loglike}"
        #assert (torch.min(new_x - self._lower).item() >= 0) and (torch.max(new_x - self._upper).item() <= 0)
        sample = NSPoints(self.nparams)
        sample.add_samples(values=new_x,
                           logL=new_loglike,
                           logweights=torch.zeros(new_loglike.shape[0], device=self.device))
        return sample


    def find_new_sample_batch(self, min_like, n_points, labels=None):
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
        newsamples = self.reflect_sampling(min_like, labels=labels)

        return newsamples

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
        newsamples = self.reflect_sampling(min_like)

        return newsamples

    def move_one_step(self):
        ''' Find highest log likes, get rid of those point, and sample a new ones '''
        if self.n_clusters == 1:
            n_points = self.nlive_ini//2
            for _ in range(n_points):
                sample = self.kill_point()

            logl = sample.get_logL().clone()

            self.add_point_batch(min_logL=logl, n_points=n_points, labels=torch.zeros(n_points, dtype=torch.int))
        else:
            new_labels = torch.zeros(self.n_clusters, dtype=torch.int)
            sample = self.kill_point()
            cluster_volumes = torch.exp(self.summaries.get_logXp())
            num_points = self.live_points.count_labels()
            idx = torch.multinomial(cluster_volumes, 1)
            new_labels[idx] += 1
            while torch.min(num_points - new_labels) > 1:
                sample = self.kill_point()
                cluster_volumes = torch.exp(self.summaries.get_logXp())
                num_points = self.live_points.count_labels()
                idx = torch.multinomial(cluster_volumes, 1)
                new_labels[idx] += 1

            logl = sample.get_logL().clone()
            self.add_point_batch(min_logL=logl, n_points=torch.sum(new_labels).item(), labels=new_labels)


if __name__ == "__main__":
    ndims = 64
    mvn1 = torch.distributions.MultivariateNormal(loc=2*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-2*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.2*torch.ones(ndims)))

    true_samples = torch.cat([mvn1.sample((5000,)), mvn2.sample((5000,))], dim=0)
    #true_samples = mvn1.sample((5000,))

    def get_loglike(theta):
        if len(theta.shape) == 1:
            theta = theta.reshape(1, -1)
        mask = (torch.min(theta, dim=-1)[0] >= -5 * torch.ones(theta.shape[0])) * ((torch.max(theta, dim=-1)[0] <= 5 * torch.ones(theta.shape[0])))
        lp = torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=0, keepdim=False) - torch.log(torch.tensor(2.0))
        #lp = mvn1.log_prob(theta) #- 1 * (~mask).float() * torch.sum(theta**2, dim=-1)#.reshape(-1, 1)
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
        dt_ini=.5,
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
    samples.saveAsText(f'2modes_dim{ndim}.txt')
    g = plots.get_subplot_plotter()
    g.triangle_plot([true_samples, samples], [f'p{i}' for i in range(5)], filled=True, legend_labels=['True', 'GDNest'])
    g.export('test_dygalilean.png')
