import torch
from gradNS.nested_sampling import NestedSampler
from gradNS.param import Param, NSPoints

# Default floating point type
dtype = torch.float64


class HamiltonianStaticNS(NestedSampler):
    """
    This Nested Sampler uses Slice Sampling to generate new samples.
    """
    def __init__(self, loglike, params, nlive=50, tol=0.1, dt_ini=0.1, num_repeats=20, verbose=True, clustering=False,
                 device=None):
        super().__init__(loglike, params, nlive, tol, verbose=verbose, clustering=clustering, device=device)
        """ 
        Parameters
        ----------
        loglike : function
            The log-likelihood function
        params : list
            A list of Param objects
        nlive : int
            The number of live points
        tol : float
            The tolerance for the stopping criterion
        verbose : bool
            Whether to print information
        num_repeats : int
            The number of repeats for the slice sampling algorithm
        clustering : bool
            Whether to use clustering
        device : torch.device
            The device to use        
        """

        self.n_repeats = 2 * int(self.nparams) if num_repeats is None else num_repeats
        self.dt = dt_ini
        self.n_in = 0
        self.n_out = 0


    def slice_sampling(self, log_slice_height, initial_x):
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

        for _ in range(self.n_repeats):
            x += velocity * self.dt

            # Check if the point is inside the prior
            in_prior = self.is_in_prior(x)
            # Calculate the log-likelihood and its gradient
            p_x, grad_p_x = self.get_score(x)

            # Check if the point is inside the slice
            reflected = p_x <= log_slice_height

            outside = reflected + ~in_prior

            if outside:
                normal = grad_p_x / torch.norm(grad_p_x, dim=1, keepdim=True)
                normal = normal.to(dtype)
                delta_velocity = 2 * torch.einsum('ai, ai -> a', velocity, normal).reshape(-1, 1) * normal
                velocity -= delta_velocity
                self.n_out += 1
            else:
                r = torch.randn_like(velocity, dtype=dtype, device=self.device)
                r /= torch.linalg.norm(r, dim=-1, keepdim=True)
                velocity = velocity * (1 + 0.05 * r)
                self.n_in += 1

        return x, p_x

    def find_new_sample(self, min_like):
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
        cluster_volumes = torch.exp(self.summaries.get_logXp())
        initial_point = self.live_points.get_random_sample(cluster_volumes)
        curr_value = initial_point.get_values()#[0]

        accepted = False
        while not accepted:
            new_value, new_loglike = self.slice_sampling(min_like, curr_value)
            frac_out = self.n_out / (self.n_in + self.n_out)

            if frac_out > 0.3:
                self.dt *= 0.5
                if self.verbose: print('dt decreased to', self.dt)
            elif frac_out < 0.1:
                self.dt *= 1.5
                if self.verbose: print('dt increased to', self.dt)

            accepted = new_loglike > min_like and self.is_in_prior(new_value)

        sample = NSPoints(self.nparams, device=self.device)
        sample.add_samples(values=new_value.reshape(1, -1),
                           logL=new_loglike.reshape(1),
                           logweights=torch.zeros(1, device=self.device),
                           labels=initial_point.get_labels())

        return sample
