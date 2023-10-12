import torch
from gradNS.nested_sampling import NestedSampler
from gradNS.param import Param, NSPoints
from continuousGFN import ContinuousGFN
from numpy import pi

# Default floating point type
dtype = torch.float64


class GaussianPolicy(torch.nn.Module):
    def __init__(self, dim):
        super(GaussianPolicy, self).__init__()
        self.t_model = torch.nn.Sequential(
            torch.nn.Linear(128,64,dtype=dtype),torch.nn.GELU(),
            torch.nn.Linear(64,64,dtype=dtype),torch.nn.GELU()
        )
        self.x_model = torch.nn.Sequential(
            torch.nn.Linear(dim,64,dtype=dtype),torch.nn.GELU(),
            torch.nn.Linear(64,64,dtype=dtype),torch.nn.GELU()
        )
        self.joint_model = torch.nn.Sequential(
            torch.nn.Linear(128,64,dtype=dtype), torch.nn.GELU(),
            torch.nn.Linear(64,64,dtype=dtype),torch.nn.GELU(),
            torch.nn.Linear(64,2*dim,dtype=dtype)
        )
        self.harmonics = torch.nn.Parameter(torch.arange(1,64+1,dtype=dtype).float() * 2 * pi).requires_grad_(False)

    def forward(self, x, t):
        t_fourier1 = (t.unsqueeze(1) * self.harmonics).sin()
        t_fourier2 = (t.unsqueeze(1) * self.harmonics).cos()
        t_emb = self.t_model(torch.cat([t_fourier1, t_fourier2], 1).to(dtype))
        x_emb = self.x_model(x.to(dtype))
        return self.joint_model(torch.cat([t_emb,x_emb],1))


class GFNNestedSampler(NestedSampler):
    """
    This Nested Sampler uses Dynamic Nested Sampling to sample the posterior.
    """
    def __init__(self, loglike, params, nlive=50, tol=0.1, rejection_fraction=0.1, clustering=False, verbose=True,
                 device=None):
        """ In this Nested Sampler, we start with a set of live points, and instead of killing one at a time, we
        kill half of them and replace them with new samples from the prior. This is done until the tolerance is reached.
        """
        super().__init__(loglike, params, nlive, tol, clustering, verbose, device)
        self.rejection_fraction = rejection_fraction
        self.has_gfn = False
        self.n_accepted_gfn = 0
        self.n_tried_gfn = 0
        self.frac_accepted = 1.
        self.gfn = None

        fwd_policy = GaussianPolicy(self.nparams)
        self.gfn = ContinuousGFN(algo='mle',
                                 t_scale=5.,
                                 dim=self.nparams,
                                 log_reward=None,
                                 fwd_policy=fwd_policy)

    def move_one_step(self):
        """
        Move one step in the Dynamic Nested Sampling algorithm, by replacing half the samples with a new half

        If using clustering, we kill points and assign a label, then add new points with the same label.
        """
        if not self.has_gfn:
            sample = self.kill_point()
            self.add_point(min_logL=sample.get_logL())
            self.n_tried_gfn = self.n_tried
            self.n_accepted_gfn += 1
            self.frac_accepted = self.n_accepted_gfn / self.n_tried_gfn

        else:
            min_like = self.live_points.get_logL().min()

            new_points = self.gfn.sample(self.nlive_ini)
            curr_likes = self.loglike(new_points)
            self.n_tried_gfn += self.nlive_ini
            new_accepted = torch.sum(curr_likes > min_like)
            self.n_accepted += new_accepted
            self.n_accepted_gfn += new_accepted

            self.frac_accepted = self.n_accepted_gfn / self.n_tried_gfn

            live_points = torch.cat([self.live_points.get_values(), new_points], dim=0)
            loglikes = torch.cat([self.live_points.get_logL(), curr_likes], dim=0)

            loglikes, idx = torch.sort(loglikes)
            live_points = live_points[idx]

            loglikes = loglikes[-self.nlive_ini:]
            live_points = live_points[-self.nlive_ini:]

            for _ in range(new_accepted):
                self.kill_point()

            self.live_points = NSPoints(nparams=self.nparams)
            self.live_points.add_samples(values=live_points, logL=loglikes, logweights=torch.zeros(self.nlive_ini, dtype=dtype, device=self.device))

            # TODO: This isnt quite right, I should only change the logL_birth of the new points
            ##self.live_points.logL_birth [self.live_points.logL_birth == 1] = min_like * torch.ones(new_accepted, dtype=dtype)
            #print(self.xlogL.shape, min_like.shape, self.summaries.get_logX().shape)
            self.xlogL = torch.cat((self.xlogL, min_like.reshape(1) + self.summaries.get_logX()))

            #print(f'Accepted {new_accepted} points. Min loglike {loglikes[0]:.4f}, fraction = {self.frac_accepted:.4f}')

        if self.frac_accepted < self.rejection_fraction:
            # FIT GFN
            # TODO: Adjust t_scale
            self.has_gfn = True
            self.n_accepted_gfn = 0
            self.n_tried_gfn = 0
            self.frac_accepted = 1.

            self.gfn.train(batch_size=self.nlive_ini, samples=self.live_points.get_values(), patience=50, num_epochs=1000)
