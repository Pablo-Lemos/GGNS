import torch
from gradNS.dynamic import DynamicNestedSampler
from gradNS.param import Param, NSPoints
from numpy import clip, pi
import tracemalloc
import gc
import pickle

# Default floating point type
dtype = torch.float32


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


    def save(self, filename):
        """
        Save the current state of the sampler (including dt for the Hamiltonian NS)
        Parameters
        ----------
        filename: str
          The name of the file to save the sampler state to
        """

        d = {'dead_points': self.dead_points,
             'live_points': self.live_points,
             'like_evals': self.like_evals,
             'n_accepted': self.n_accepted,
             'cluster_volumes': self.cluster_volumes,
             'n_clusters': self.n_clusters,
             'xlogL': self.xlogL,
             'summaries': self.summaries,
             'dt': self.dt}

        with open(filename, 'wb') as f:
            pickle.dump(d, f)

    def slice_sampling(self, log_slice_height, initial_x, dts):
        """
        Slice sampling algorithm for PyTorch.

        Parameters
        ----------
        log_slice_height : float
            The height of the slice
        initial_x : torch.Tensor
            The initial point

        Returns
        -------
        x : torch.Tensor
            The new sample
        """

        x = initial_x.clone()
        # Choose a random direction
        d = torch.randn_like(x, dtype=dtype, device=self.device)
        velocity = d / torch.linalg.norm(d, dim=1, keepdim=True)

        pts = torch.zeros((0, self.nparams), dtype=dtype, device=self.device)
        min_reflections = 3
        max_reflections = 5
        num_reflections = torch.zeros(x.shape[0], dtype=torch.int, device=self.device)

        active = torch.ones(x.shape[0], dtype=torch.bool, device=self.device)
        n_in = torch.zeros(x.shape[0], dtype=torch.int, device=self.device)
        n_out = torch.zeros(x.shape[0], dtype=torch.int, device=self.device)

        final_x = torch.zeros_like(x, dtype=dtype, device=self.device)
        final_logl = -1e100 * torch.ones(x.shape[0], dtype=dtype, device=self.device)

        num_tries = torch.ones(x.shape[0], dtype=torch.int, device=self.device)
        num_steps = 0
        #while num_reflections < max_reflections:
        while (torch.sum(active) > 0):#and (num_steps < 1000):
            x += velocity * dts.reshape(-1, 1)

            # Check if the point is inside the prior
            in_prior = self.is_in_prior(x)
            # Calculate the log-likelihood and its gradient
            p_x, grad_p_x = self.get_score(x)

            # Check if the point is inside the slice
            p_x = p_x.to(self.device)
            reflected = p_x <= log_slice_height

            outside = reflected + ~in_prior

            normal = grad_p_x[outside] / torch.norm(grad_p_x[outside], dim=1, keepdim=True)
            normal = normal.to(dtype)
            delta_velocity = 2 * torch.einsum('ai, ai -> a', velocity[outside], normal).reshape(-1, 1) * normal
            velocity[outside] -= delta_velocity
            n_out[outside] += 1
            num_reflections[outside] += 1

            r = torch.randn_like(velocity[~outside], dtype=dtype, device=self.device)
            r /= torch.linalg.norm(r, dim=-1, keepdim=True)
            velocity[~outside] = velocity[~outside] * (1 + 0.05 * r)
            n_in[~outside] += 1

            store = (num_reflections >= min_reflections) * ~outside
            for i in range(x.shape[0]):
                if store[i]:
                    # Store the point with probability 1 / num_tries
                    if torch.rand(1) < (1 / num_tries[i]):
                        final_x[i] = x[i]
                        assert p_x[i] > log_slice_height, f"p_x = {p_x[i]}, log_slice_height = {log_slice_height}"
                        final_logl[i] = p_x[i]

                    num_tries[i] += 1

            active = num_reflections < max_reflections
            num_steps += 1

        return final_x, final_logl, n_out, n_in

    def find_new_sample_batch(self, min_loglike, n_points, labels=None):
        """
        Use slice sampling to find a new sample.

        Parameters
        ----------
        min_like : float
            The minimum likelihood value

        Returns
        -------
        sample : NSPoints
            The new sample
        """
        point = self.live_points.get_samples_from_labels(labels)
        ini_labels = point.get_labels() # bincount equal to labels
        curr_values = point.get_values()

        # Initalize arrays
        active = torch.ones(curr_values.shape[0], dtype=torch.bool, device=self.device)
        dts = self.dt * torch.ones(curr_values.shape[0], dtype=dtype, device=self.device)

        new_value = torch.zeros_like(curr_values, dtype=dtype, device=self.device)
        new_loglike = torch.zeros(curr_values.shape[0], dtype=dtype, device=self.device)
        n_out = torch.zeros(curr_values.shape[0], dtype=torch.int, device=self.device)
        n_in = torch.zeros(curr_values.shape[0], dtype=torch.int, device=self.device)

        while torch.sum(active) > 0:
            new_value[active], new_loglike[active], n_out[active], n_in[active] = self.slice_sampling(min_loglike, curr_values[active], dts[active])
            frac_out = n_out / (n_in + n_out)

            frac_out[~active] = 0.3
            mask_increase = frac_out < 0.1
            mask_decrease = frac_out > 0.5

            dts[mask_increase] *= 1.5
            dts[mask_decrease] *= 0.5

            new_loglike = new_loglike.to(self.device)
            accepted = (new_loglike > min_loglike) * self.is_in_prior(new_value)

            active = ~accepted

        self.dt = torch.mean(dts).item()
        sample = NSPoints(self.nparams, device=self.device)
        sample.add_samples(values=new_value,
                           logL=new_loglike,
                           logweights=torch.zeros(new_value.shape[0], dtype=dtype, device=self.device),
                           labels=point.get_labels())

        return sample


#
#     def find_new_sample_batch(self, min_loglike, n_points, labels=None):
#         """
#         Sample the prior until finding a sample with higher likelihood than a
#         given value
#         Parameters
#         ----------
#         min_like : float
#            The threshold log-likelihood
#         Returns
#         -------
#         newsample : pd.DataFrame
#            A new sample
#         """
#         # Get initial points from set of existing live points
#         # n_samples_per_label = torch.bincount(labels)
#
#         point = self.live_points.get_samples_from_labels(labels)
#         ini_labels = point.get_labels() # bincount equal to labels
#         x_ini = point.get_values()
#
#         # Initalize arrays
#         active = torch.ones(x_ini.shape[0], dtype=torch.bool, device=self.device)
#         new_x = torch.zeros_like(x_ini, dtype=dtype, device=self.device)
#         new_loglike = torch.zeros(x_ini.shape[0], dtype=dtype, device=self.device)
#
#         accepted = False
#         while not accepted:
#             assert torch.min(
#                 self.loglike(x_ini)) >= min_loglike, f"min_loglike = {min_loglike}, x_loglike = {self.loglike(x_ini)}"
#             assert self.is_in_prior(x_ini).all(), f"min_loglike = {min_loglike}, x_loglike = {self.loglike(x_ini)}"
#
#             # Initialize velocity randomly
#             velocity = torch.randn_like(x_ini, dtype=dtype, device=self.device)
#             velocity /= torch.linalg.norm(velocity, dim=-1, keepdim=True)
#
#             new_x_active, new_loglike_active, out_frac = self.hamiltonian_slice_sampling(position=x_ini[active],
#                                                                                          velocity=velocity[active],
#                                                                                          min_like=min_loglike)
#             new_x[active] = new_x_active
#             new_loglike[active] = new_loglike_active
#
#             # Adapt time step if there are too many, ot not enough reflections
# #            if (out_frac > 0.1) and (torch.sum(active).item() > len(active) // 2):
#             if (out_frac > 0.15) and (torch.sum(active).item() >= max(2, len(active) // 2)):
#                 self.dt = clip(self.dt * 0.9, 1e-5, 10)
#                 #self.dt = self.dt * 0.9
#                 if self.verbose: print("Decreasing dt to ", self.dt,
#                                        "out_frac = ", out_frac, "active = ", torch.sum(active).item())
#                 active = torch.ones(x_ini.shape[0], dtype=torch.bool, device=self.device)
#             elif (out_frac < 0.05) and (torch.sum(active).item() >= max(2, len(active) // 2)):
#             #elif (out_frac < 0.01) and (torch.sum(active).item() > len(active) // 2):
#                 self.dt = clip(self.dt * 1.1, 1e-5, 10)
#                 if self.verbose: print("Increasing dt to ", self.dt,
#                                        "out_frac = ", out_frac, "active = ", torch.sum(active).item())
#                 active = torch.ones(x_ini.shape[0], dtype=torch.bool, device=self.device)
#             else:
#                 in_prior = self.is_in_prior(new_x)
#                 # Count the number of points that have not been accepted
#                 active = (new_loglike < min_loglike) + (~in_prior)
#                 if self.verbose and torch.sum(active) > 0:
#                     print(f"Active: {torch.sum(active).item()} / {len(active)}")
#
#                 del in_prior
#
#             accepted = torch.sum(active) == 0
#
#         assert torch.min(new_loglike) >= min_loglike, f"min_loglike = {min_loglike}, new_loglike = {new_loglike}"
#         sample = NSPoints(self.nparams, device=self.device)
#         sample.add_samples(values=new_x,
#                            logL=new_loglike,
#                            logweights=torch.zeros(new_loglike.shape[0], device=self.device),
#                            labels=ini_labels
#                            )
#
#         del new_x
#         del new_loglike
#         del active
#         del out_frac
#         del point
#         del x_ini
#
#         gc.collect()
#
#         return sample
#
