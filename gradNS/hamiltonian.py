import torch
from gradNS.dynamic import DynamicNestedSampler
from gradNS.param import Param, NSPoints
from numpy import clip, pi
import tracemalloc
import gc

# Default floating point type
dtype = torch.float64


class HamiltonianNS(DynamicNestedSampler):
    """
    This Nested Sampler uses Dynamic Hamiltonian Slice Sampling.
    """
    def __init__(self, loglike, params, nlive=50, tol=0.1, dt_ini=0.1, min_reflections=1, max_reflections=3,
                 sigma_vel=0.05, rejection_fraction=0.1, clustering=False, verbose=True, device=None):
        super().__init__(loglike, params, nlive, tol, rejection_fraction, clustering, verbose, device)

        # Initial time step size (it will be adapted)
        self.dt = dt_ini

        # Minimum and maximum number of reflections
        self.min_reflections = min_reflections
        self.max_reflections = max_reflections

        # Standard deviation for the velocity noise
        self.sigma_vel = sigma_vel

    def hamiltonian_slice_sampling(self, position, velocity, min_like):
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
        """
        assert(len(position.shape) == 2), "Position must be a 2D tensor"
        # Keep track of the number of reflections and inside steps, to adjust the time step
        n_out_steps = 0
        n_in_steps = 0
        # Keep track of the number of reflections for each point
        num_reflections = torch.zeros(position.shape[0], dtype=torch.int64, device=self.device)

        # A list of positions, log-likelihoods and masks for each point
        # pos_ls = []
        # logl_ls = []
        # mask = []

        pos_tensor = torch.zeros((0, position.shape[0], self.nparams), dtype=dtype, device=self.device)
        logl_tensor = torch.zeros((0, position.shape[0]), dtype=dtype, device=self.device)
        mask_tensor = torch.zeros((0, position.shape[0]), dtype=torch.bool, device=self.device)

        # A list of the reflections
        memory = torch.zeros((position.shape[0], 3), dtype=torch.bool, device=self.device)

        # The initial position is the first point
        x = position.clone()

        killed = torch.zeros(position.shape[0], dtype=torch.bool, device=self.device)

        original_indices = list(range(position.shape[0]))  # Create a list of original indices
        killed_indices = []  # Create an empty list to store the indices of killed points

        num_steps = 0
        start_saving = False
        # The algorithm stops when the point with the smallest number of reflections has reached the
        # maximum number of reflections
        while (torch.min(num_reflections) < self.max_reflections) and (num_steps < self.max_reflections * 100):
            num_steps += 1
            # Update position with Euler step
            x += velocity * self.dt

            # Check if the point is inside the prior
            in_prior = self.is_in_prior(x)
            # Calculate the log-likelihood and its gradient
            p_x, grad_p_x = self.get_score(x)

            x = x[~killed]
            p_x = p_x[~killed]
            grad_p_x = grad_p_x[~killed]
            in_prior = in_prior[~killed]
            velocity = velocity[~killed]
            num_reflections = num_reflections[~killed]
            memory = memory[~killed]
            pos_tensor = pos_tensor[:, ~killed]
            logl_tensor = logl_tensor[:, ~killed]
            mask_tensor = mask_tensor[:, ~killed]
            killed = killed[~killed]

            if len(x) == 0:
                return torch.zeros_like(position, dtype=dtype, device=self.device), \
                       -1e30 * torch.ones(position.shape[0], dtype=dtype, device=self.device), 1

            # Check if the point is inside the slice
            reflected = p_x <= min_like

            outside = reflected + ~in_prior
            memory[:, 0] = memory[:, 1]
            memory[:, 1] = memory[:, 2]
            memory[:, 2] = outside

            # Kill the points that have been reflected more than 3 times in a row
            killed = torch.sum(memory, dim=1) == 3

            # Get the indices of killed points and remove them from the original_indices list
            killed_idx = torch.where(killed)[0].tolist()
            killed_indices.extend([original_indices[idx] for idx in killed_idx])
            original_indices = [i for j, i in enumerate(original_indices) if j not in killed_idx]

            # If the point is inside the slice, update the velocity
            normal = grad_p_x / torch.norm(grad_p_x, dim=1, keepdim=True)
            normal = normal.to(dtype)
            delta_velocity = 2 * torch.einsum('ai, ai -> a', velocity, normal).reshape(-1, 1) * normal
            velocity[outside, :] -= delta_velocity[outside, :]

            # Update the number of positions
            num_reflections += reflected
            if torch.min(num_reflections) > self.min_reflections:
                start_saving = True
                pos_tensor = torch.cat((pos_tensor, x.unsqueeze(0).clone()), dim=0)
                logl_tensor = torch.cat((logl_tensor, p_x.clone().reshape(1, -1)), dim=0)
                mask_tensor = torch.cat((mask_tensor, ~outside.clone().reshape(1, -1)), dim=0)

            if self.prior is not None:
                r = torch.randn_like(velocity[~outside], dtype=dtype, device=self.device)
                velocity[~outside] = self.dt * self.prior(x[~outside]) + 2 ** 0.5 * r

            if self.sigma_vel > 0:
                r = torch.randn_like(velocity[~reflected * in_prior], dtype=dtype, device=self.device)
                r /= torch.linalg.norm(r, dim=-1, keepdim=True)
                velocity[~outside] = velocity[~outside] * (1 + self.sigma_vel * r)

            # Update the number of points inside an outside
            #n_out_steps += (outside * ~killed).sum()
            #n_in_steps += (~outside * ~killed).sum()
            n_out_steps += outside.sum()
            n_in_steps += (~outside).sum()

        # Fraction of points outside the slice
        out_frac = n_out_steps / (n_out_steps + n_in_steps)

        if not start_saving:
            x = x[~killed]
            p_x = p_x[~killed]
            num_reflections = num_reflections[~killed]
            pos_tensor = pos_tensor[:, ~killed]
            logl_tensor = logl_tensor[:, ~killed]
            mask_tensor = mask_tensor[:, ~killed]
            killed = killed[~killed]

            killed = num_reflections < self.min_reflections
            # Get the indices of killed points and remove them from the original_indices list
            killed_idx = torch.where(killed)[0].tolist()
            killed_indices.extend([original_indices[idx] for idx in killed_idx])
            original_indices = [i for j, i in enumerate(original_indices) if j not in killed_idx]

            x = x[~killed]
            p_x = p_x[~killed]
            pos_tensor = pos_tensor[:, ~killed]
            logl_tensor = logl_tensor[:, ~killed]
            mask_tensor = mask_tensor[:, ~killed]

            pos_tensor = torch.cat((pos_tensor, x.unsqueeze(0).clone()), dim=0)
            logl_tensor = torch.cat((logl_tensor, p_x.clone().reshape(1, -1)), dim=0)
            mask_tensor = torch.cat((mask_tensor, torch.ones(1, x.shape[0], dtype=torch.bool, device=self.device)), dim=0)


        # If no point has reached the minimum number of reflections, return a point with zero likelihood
        if pos_tensor.shape[0] == 0:
            pos_out = torch.zeros_like(x, dtype=dtype, device=self.device)
            logl_out = torch.tensor(-1e30, dtype=torch.float64) * torch.ones(position.shape[0], dtype=dtype, device=self.device)
        # Otherwise, return a random point from the list
        else:
            # positions = torch.stack(pos_ls, dim=0)
            # loglikes = torch.stack(logl_ls, dim=0)
            # masks = torch.stack(mask, dim=0)
            pos_out = torch.zeros(position.shape, dtype=dtype, device=self.device)
            logl_out = torch.zeros(position.shape[0], dtype=dtype, device=self.device)


            k = 0
            for i in range(position.shape[0]):
                if i in original_indices:
                    pos = pos_tensor[:, k, :]
                    ll = logl_tensor[:, k]

                    if torch.sum(mask_tensor[:, k]) == 0:
                        pos_out[i, :] = 0.
                        logl_out[i] = -1e30
                    else:
                        pos = pos[mask_tensor[:, k]]
                        ll = ll[mask_tensor[:, k]]
                        idx = torch.randint(0, pos.shape[0], (1,))
                        pos_out[i, :] = pos[idx, :].clone()
                        logl_out[i] = ll[idx].clone()
                    k += 1
                else:
                    pos_out[i, :] = 0.
                    logl_out[i] = -1e30

        del delta_velocity
        del grad_p_x
        del in_prior
        del killed
        del logl_tensor
        # del loglikes
        # del mask
        # del masks
        del mask_tensor
        del memory
        del n_in_steps
        del n_out_steps
        del normal
        del num_reflections
        del outside
        del p_x
        # del pos_ls
        del pos_tensor
        del position
        del reflected
        del velocity
        del x

        gc.collect()

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
        # n_samples_per_label = torch.bincount(labels)

        point = self.live_points.get_samples_from_labels(labels)
        ini_labels = point.get_labels() # bincount equal to labels
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

            new_x_active, new_loglike_active, out_frac = self.hamiltonian_slice_sampling(position=x_ini[active],
                                                                                         velocity=velocity[active],
                                                                                         min_like=min_loglike)
            new_x[active] = new_x_active
            new_loglike[active] = new_loglike_active

            # Adapt time step if there are too many, ot not enough reflections
#            if (out_frac > 0.1) and (torch.sum(active).item() > len(active) // 2):
            if (out_frac > 0.15) and (torch.sum(active).item() >= max(2, len(active) // 2)):
                self.dt = clip(self.dt * 0.5, 1e-5, 10)
                if self.verbose: print("Decreasing dt to ", self.dt,
                                       "out_frac = ", out_frac, "active = ", torch.sum(active).item())
                active = torch.ones(x_ini.shape[0], dtype=torch.bool)
            elif (out_frac < 0.05) and (torch.sum(active).item() >= max(2, len(active) // 2)):
            #elif (out_frac < 0.01) and (torch.sum(active).item() > len(active) // 2):
                self.dt = clip(self.dt * 1.5, 1e-5, 10)
                if self.verbose: print("Increasing dt to ", self.dt,
                                       "out_frac = ", out_frac, "active = ", torch.sum(active).item())
                active = torch.ones(x_ini.shape[0], dtype=torch.bool)
            else:
                in_prior = self.is_in_prior(new_x)
                # Count the number of points that have not been accepted
                active = (new_loglike < min_loglike) + (~in_prior)
                if self.verbose and torch.sum(active) > 0:
                    print(f"Active: {torch.sum(active).item()} / {len(active)}")

                del in_prior

            accepted = torch.sum(active) == 0

        assert torch.min(new_loglike) >= min_loglike, f"min_loglike = {min_loglike}, new_loglike = {new_loglike}"
        sample = NSPoints(self.nparams, device=self.device)
        sample.add_samples(values=new_x,
                           logL=new_loglike,
                           logweights=torch.zeros(new_loglike.shape[0], device=self.device),
                           labels=ini_labels
                           )

        del new_x
        del new_loglike
        del active
        del out_frac
        del point
        del x_ini

        #gc.collect()

        return sample

