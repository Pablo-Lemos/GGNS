import torch
from torchNS.dynamic import DynamicNestedSampler
from torchNS.param import Param, NSPoints
from numpy import clip, pi

# Default floating point type
dtype = torch.float64


class HamiltonianNS(DynamicNestedSampler):
    """
    This Nested Sampler uses Dynamic Hamiltonian Slice Sampling.
    """
    def __init__(self, loglike, params, nlive=50, tol=0.1, dt_ini=0.1, clustering=False, verbose=True, device=None):
        super().__init__(loglike, params, nlive, tol, clustering, verbose, device)

        # Initial time step size (it will be adapted)
        self.dt = dt_ini

    def hamiltonian_slice_sampling(self, position, velocity, min_like, min_reflections, max_reflections):
        """
        Hamiltonian Slice Sampling algorithm for PyTorch.

        Parameters
        ----------
        position : torch.Tensor
            The initial position
        velocity : torch.Tensor
            The initial velocity
        min_like : float
            The minimum likelihood
        min_reflections : int
            The minimum number of reflections
        max_reflections : int
            The maximum number of reflections
        """
        assert(len(position.shape) == 2), "Position must be a 2D tensor"
        # Keep track of the number of reflections and inside steps, to adjust the time step
        n_out_steps = 0
        n_in_steps = 0
        # Keep track of the number of reflections for each point
        num_reflections = torch.zeros(position.shape[0], dtype=torch.int64, device=self.device)

        # A list of positions, log-likelihoods and masks for each point
        pos_ls = []
        logl_ls = []
        mask = []

        # The initial position is the first point
        x = position.clone()

        # The algorithm stops when the point with the smallest number of reflections has reached the
        # maximum number of reflections
        while torch.min(num_reflections) < max_reflections:
            # Update position with Euler step
            x += velocity * self.dt

            # Check if the point is inside the prior
            in_prior = self.is_in_prior(x)
            # Calculate the log-likelihood and its gradient
            p_x, grad_p_x = self.get_score(x)

            # Check if the point is inside the slice
            reflected = (p_x <= min_like)

            # If the point is inside the slice, update the velocity
            normal = grad_p_x / torch.norm(grad_p_x, dim=1, keepdim=True)
            normal = normal.to(dtype)
            delta_velocity = 2 * torch.einsum('ai, ai -> a', velocity, normal).reshape(-1, 1) * normal
            velocity[reflected + ~in_prior, :] -= delta_velocity[reflected + ~in_prior, :]

            # Update the number of positions
            num_reflections += reflected.clone()
            if torch.min(num_reflections) > min_reflections:
                # If the point has reached the minimum number of reflections, save it
                pos_ls.append(x.clone())
                logl_ls.append(p_x.clone())
                mask.append(~reflected * in_prior)

            # Update the number of points inside an outside
            n_out_steps += reflected.sum()
            n_in_steps += (~reflected).sum()

        # Fraction of points outside the slice
        out_frac = n_out_steps / (n_out_steps + n_in_steps)

        # If no point has reached the minimum number of reflections, return a point with zero likelihood
        if len(pos_ls) == 0:
            pos_out = torch.zeros_like(x)
            logl_out = torch.tensor(-1e30, dtype=torch.float64) * torch.ones(position.shape[0], dtype=dtype, device=self.device)
        # Otherwise, return a random point from the list
        else:
            positions = torch.stack(pos_ls, dim=0)
            loglikes = torch.stack(logl_ls, dim=0)
            masks = torch.stack(mask, dim=0)
            pos_out = torch.zeros(position.shape, dtype=dtype, device=self.device)
            logl_out = torch.zeros(position.shape[0], dtype=dtype, device=self.device)

            for i in range(positions.shape[1]):
                pos = positions[:, i, :]
                ll = loglikes[:, i]

                if torch.sum(masks[:, i]) == 0:
                    pos_out[i, :] = 0.
                    logl_out[i] = -1e30
                else:
                    pos = pos[masks[:, i]]
                    ll = ll[masks[:, i]]
                    idx = torch.randint(0, pos.shape[0], (1,))
                    pos_out[i, :] = pos[idx, :].clone()
                    logl_out[i] = ll[idx].clone()

        return pos_out, logl_out, out_frac

    def find_new_sample_batch(self, min_loglike, n_points, labels=None):
        """
        Sample the prior until finding a sample with higher likelihood than a
        given value
        Parameters
        ----------
        min_like : float
           The threshold log-likelihood
        Returns
        -------
        newsample : pd.DataFrame
           A new sample
        """
        # Get initial points from set of existing live points
        n_samples_per_label = torch.bincount(labels)
        point = self.live_points.get_samples_from_labels(n_samples_per_label)
        x_ini = point.get_values()

        # Initalize arrays
        active = torch.ones(x_ini.shape[0], dtype=torch.bool)
        new_x = torch.zeros_like(x_ini, dtype=dtype)
        new_loglike = torch.zeros(x_ini.shape[0], dtype=dtype)

        accepted = False
        while not accepted:
            assert torch.min(
                self.loglike(x_ini)) >= min_loglike, f"min_loglike = {min_loglike}, x_loglike = {self.loglike(x_ini)}"
            assert self.is_in_prior(x_ini).all(), f"min_loglike = {min_loglike}, x_loglike = {self.loglike(x_ini)}"

            # Initialize velocity randomly
            velocity = torch.randn_like(x_ini, dtype=dtype, device=self.device)
            velocity /= torch.linalg.norm(velocity, dim=-1, keepdim=True)

            # Run Hamiltonian slice sampling
            new_x_active, new_loglike_active, out_frac = self.hamiltonian_slice_sampling(position=x_ini[active],
                                                                                         velocity=velocity[active],
                                                                                         min_like=min_loglike,
                                                                                         min_reflections=5,
                                                                                         max_reflections=10,
                                                                                         )
            new_x[active] = new_x_active#.to(dtype)
            new_loglike[active] = new_loglike_active#.to(dtype)

            # Adapt time step if there are too many, ot not enough reflections
            if (out_frac > 0.2) and (torch.sum(active).item() > len(active) // 2):
                self.dt = clip(self.dt * 0.9, 1e-5, 10)
                if self.verbose: print("Decreasing dt to ", self.dt)
                active = torch.ones(x_ini.shape[0], dtype=torch.bool)
            elif (out_frac < 0.05) and (torch.sum(active).item() > len(active) // 2):
                self.dt = clip(self.dt * 1.1, 1e-5, 10)
                if self.verbose: print("Increasing dt to ", self.dt)
                active = torch.ones(x_ini.shape[0], dtype=torch.bool)
            else:
                in_prior = self.is_in_prior(new_x)
                # Count the number of points that have not been accepted
                active = (new_loglike < min_loglike) + (~in_prior)

            accepted = torch.sum(active) == 0
            # if not accepted:
            #     if self.verbose: print(f"Active: {torch.sum(active).item()} / {len(active)}")

        assert torch.min(new_loglike) >= min_loglike, f"min_loglike = {min_loglike}, new_loglike = {new_loglike}"
        #if self.verbose: print(f"ACCEPTED")
        sample = NSPoints(self.nparams)
        sample.add_samples(values=new_x,
                           logL=new_loglike,
                           logweights=torch.zeros(new_loglike.shape[0], device=self.device),
                           labels=labels
                           )
        return sample

