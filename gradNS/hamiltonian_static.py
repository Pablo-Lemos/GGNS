import torch
from gradNS.nested_sampling import NestedSampler
from gradNS.param import Param, NSPoints
import pickle

# Default floating point type
dtype = torch.float32


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

    def load(self, filename):
        """
        Load the current state of the sampler (including dt for the Hamiltonian NS)
        Parameters
        ----------
        filename: str
          The name of the file to load the sampler state from
        """
        with open(filename, 'rb') as f:
            d = pickle.load(f)

        self.dead_points = d['dead_points']
        self.live_points = d['live_points']
        self.like_evals = d['like_evals']
        self.n_accepted = d['n_accepted']
        self.cluster_volumes = d['cluster_volumes']
        self.n_clusters = d['n_clusters']
        self.xlogL = d['xlogL']
        self.summaries = d['summaries']
        self.dt = d['dt']

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

        pts = torch.zeros((0, self.nparams), dtype=dtype, device=self.device)
        min_reflections = 3
        max_reflections = 5
        num_reflections = 0

        while num_reflections < max_reflections:
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
                num_reflections += 1
            else:
                r = torch.randn_like(velocity, dtype=dtype, device=self.device)
                r /= torch.linalg.norm(r, dim=-1, keepdim=True)
                velocity = velocity * (1 + 0.05 * r)
                self.n_in += 1

            if num_reflections >= min_reflections and ~outside:
                pts = torch.cat((pts, x.reshape(1, -1)), dim=0)

        # Choose a new point from the points that were inside the slice
        if pts.shape[0] == 0:
            return initial_x, log_slice_height - 1
        final_x = pts[torch.randint(pts.shape[0], (1,))]
        p_x, grad_p_x = self.get_score(final_x)
        return final_x, p_x

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
