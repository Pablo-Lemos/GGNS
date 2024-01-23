#!/usr/bin/env pypy3

'''
Base Nested Sampler class
'''

import torch
import time
from gradNS.utils import uniform, get_knn_clusters, save_to_file, read_from_file
from gradNS.param import Param, NSPoints
from gradNS.summaries import NestedSamplingSummaries
from getdist import MCSamples, plots
import anesthetic
from math import log, exp
import os
import tracemalloc
import pickle
import gc

# Default floating point type
dtype = torch.float64

class Prior():
    """
    A class to represent priors to be sampled
    """

    def __init__(self, score, sample):
        """
        Parameters
        ----------
        score : function
            The score function
        sample : function
            The sampling function
        """
        assert callable(score), 'score must be a callable function'
        assert callable(sample), 'sample must be a callable function'
        self.score = score
        self.sample = sample

    def __call__(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            The point to evaluate the prior at
        """

        return self.score(x)

    def sample(self, npoints):
        """
        Parameters
        ----------
        npoints : int
            The number of points to sample
        """

        return self.sample(npoints)


class NestedSampler:
    """
    Nested Sampling class
    """
    def __init__(
            self, loglike, params, nlive=50, tol=0.1, clustering=False, verbose=True, device=None):
        """
        Parameters
        ----------
        loglike: function
          The logarithm of the likelihood function
        params: ls[params]
          A list containing all parameters, the elements belong to the parameter
          class
        nlive : int
          The number of live points. Should be set to ~25*nparams. Defaults to 50
        tol: float
          The tolerance, which decides the stopping of the algorithm. Defaults to
          0.1
        clustering: bool
            A boolean indicating if clustering should be used. Defaults to False

        verbose: bool
            A boolean indicating if the algorithm should print information as it runs
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.loglike = loglike
        self.params = params
        self.nlive_ini = nlive
        self.tol = tol

        self.nparams = len(params)
        self.paramnames = []
        self.paramlabels = []
        self.dead_points = NSPoints(self.nparams, device=self.device)
        self.live_points = NSPoints(self.nparams, device=self.device)
        self.like_evals = 0
        self.verbose = verbose
        self.clustering = clustering
        self.n_clusters = 1
        self.summaries = NestedSamplingSummaries(device=self.device)
        self.cluster_volumes = torch.ones(self.n_clusters, device=self.device)
        self.n_accepted = 0
        self.n_tried = 0
        self.prior = None
        self.xlogL = torch.tensor([-1e30], device=self.device)

        self._lower = torch.tensor([p.prior[0] for p in self.params], dtype=dtype, device=self.device)
        self._upper = torch.tensor([p.prior[1] for p in self.params], dtype=dtype, device=self.device)

    def save(self, filename):
        """
        Save the current state of the sampler
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
             'summaries': self.summaries}

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

    def add_prior(self, prior):
        """
        Parameters
        ----------
        prior : Prior
            The prior to be added
        """
        assert isinstance(prior, Prior), 'Prior must be an instance of the Prior class'
        self.prior = prior

    def sample_prior(self, npoints, initial_step=False):
        """ Produce samples from the prior distributions
        Parameters:
        -----------
        npoints : int
          The number of samples to be produced
        initial_step : bool
          A boolean indicating if this is the initial sampling step. Defaults to
          False
        Returns:
        samples: pd.Dataframe
          A pandas dataframe containing the values of the parameters and the
          corresponding log likelihood, prior probability, and log posterior. The
          weights are set to unity, as they are calculated in a separate function
        """

        # Create an empty list for the samples
        prior_samples = torch.zeros([npoints, self.nparams], dtype=dtype, device=self.device)

        if self.prior is None:
            # Iterate over all parameters
            for i, param in enumerate(self.params):
                if initial_step == True:
                    self.paramnames.append(param.name)
                    self.paramlabels.append(param.label)
                if param.prior_type == 'Uniform':
                    prior_samples[:,i] = uniform(low=param.prior[0],
                                                 high=param.prior[1],
                                                 size=npoints,
                                                 dtype=dtype,
                                                 device=self.device)
                # elif param.prior_type == 'Gaussian':
                #     prior_samples[:,i] = torch.normal(mean=param.prior[0],
                #                                       std=param.prior[1],
                #                                       size=npoints,
                #                                       device=self.device)
                else:
                    raise ValueError('Prior type not recognised, only Uniform is implemented for now')
        else:
            if initial_step == True:
                for i, param in enumerate(self.params):
                    self.paramnames.append(param.name)
                    self.paramlabels.append(param.label)
            prior_samples = self.prior.sample(npoints)

        # Calculate log likelihood
        logL = torch.zeros(npoints, dtype=dtype, device=self.device)
        for i, sample in enumerate(prior_samples):
            logL[i] = self.loglike(sample)

        # Placeholder weights (will be calculated when the point is killed)
        logweights = torch.zeros(len(logL), dtype=dtype, device=self.device)
        points = NSPoints(self.nparams, device=self.device)
        points.add_samples(values=prior_samples, logweights=logweights, logL=logL,
                           logL_birth=-1e30*torch.ones_like(logL, dtype=dtype, device=self.device))

        # Count likelihood evaluations
        self.like_evals += npoints
        return points

    def is_in_prior(self, x):
        """ Check if a point is in the prior
        Parameters:
        -----------
        x: torch.tensor
            A 1D or 2D tensor containing the parameter values
        Returns:
        --------
        in_prior: torch.tensor
            A boolean tensor indicating if the point is in the prior
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self.nparams
        if self.prior is None:
            in_prior = (torch.min(x - self._lower, dim=-1)[0] >= torch.zeros(x.shape[0], dtype=dtype, device=self.device)) * \
                       (torch.max(x - self._upper, dim=-1)[0] <= torch.zeros(x.shape[0], dtype=dtype, device=self.device))
            return in_prior
        else:
            return torch.ones(x.shape[0], dtype=torch.bool, device=self.device)

    def get_score(self, theta):
        """ Calculate the score for a given point
        Parameters:
        -----------
        theta: torch.tensor
            A 1D or 2D tensor containing the parameter values
        Returns:
        --------
        score: torch.tensor
            A tensor containing the score for each point
        """
        if theta.dim() == 1:
            self.like_evals += 1
            theta = theta.unsqueeze(0)
        elif theta.dim() == 2:
            self.like_evals += theta.shape[0]
        else:
            raise ValueError("theta must be 1 or 2 dimensional")
        theta = theta.clone().detach().requires_grad_(True)
        loglike = self.loglike(theta)

        # Calculate the score when the point is outside the prior range
        v_low = torch.zeros_like(theta, dtype=dtype, device=self.device)
        v_high = torch.zeros_like(theta, dtype=dtype, device=self.device)
        v_low[theta < self._lower] = 1.
        v_high[theta > self._upper] = -1.

        v_ref = v_low + v_high
        v_ref = v_ref / (torch.norm(v_ref, dim=-1, keepdim=True) + 1e-10)
        in_prior = self.is_in_prior(theta)

        score = torch.autograd.grad(loglike, theta, torch.ones_like(loglike, dtype=dtype, device=self.device))[0]

        with torch.no_grad():
            score[~in_prior] = v_ref[~in_prior]

        if torch.isnan(score).any():
            raise ValueError("Score is NaN for theta = {}".format(theta))
        return loglike, score

    def get_like_evals(self):
        """ Return the number of likelihood evaluations
        Returns
        -------
        like_evals : int
            The number of likelihood evaluations
        """
        return self.like_evals

    def get_weight(self, sample):
        """ Calculate the weight at a given iteration, calculated as the number of dead points

        Parameters
        ----------
        sample : NSPoint
            The point to calculate the weight for

        Returns
        -------
        weight : float
            The weight of the point
        """

        label = sample.get_labels().item()
        try:
            n_p = self.live_points.count_labels()[label]
        except IndexError:
            n_p = 0
        logweight = self.summaries.get_logXp()[label] - torch.log(torch.as_tensor(n_p + 1, device=self.device))
        weight = torch.exp(logweight)
        return weight

    def get_nlive(self):
        """ Return the number of live points
        Returns
        -------
        nlive : int
            The number of live points
        """
        return self.live_points.get_size()

    def _get_epsilon(self):
        """ Find the maximum contribution to the evidence from the livepoints,
        used for the stopping criterion
        Returns
        -------
        delta_logZ : float
          The maximum contribution to the evidence from the live points
        """

        if not self.clustering:
            # Find index with mean likelihood
            mean_logL = torch.mean(self.live_points.get_logL())

            logXi = self.summaries.get_logXp()

            # Get delta_logZ as log(Xi*L)
            delta_logZ = logXi + mean_logL

            epsilon = torch.exp(delta_logZ - self.summaries.get_logZ())

        else:
            delta_logZ = torch.zeros(self.n_clusters, device=self.device)
            for i in range(self.n_clusters):
                # Select points in cluster
                cluster_points = self.live_points.get_cluster(i)

                if cluster_points.get_size() == 0:
                    delta_logZ[i] = -torch.inf
                    continue

                # Find index with mean likelihood
                mean_logL = torch.mean(cluster_points.get_logL())

                logXi = self.summaries.get_logXp()[i]

                # Get delta_logZ as log(Xi*L)
                delta_logZ[i] = logXi + mean_logL

            epsilon = torch.exp(delta_logZ - self.summaries.get_logZp())

        return epsilon

    def find_new_sample(self, min_like):
        """ Sample the prior until finding a sample with higher likelihood than a
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

        newlike = -torch.inf
        while newlike < min_like:
            newsample = self.sample_prior(npoints=1)
            newlike = newsample.get_logL()[0]
            self.n_tried += 1

        return newsample

    def find_clusters(self):
        """ Run a clustering algorithm to find how many clusters are present in the posterior
        """
        for i in range(self.n_clusters):
            all_labels = self.live_points.get_labels().detach().numpy()
            x = self.live_points.get_cluster(i).get_values()
            if x.shape[0] < self.nlive_ini//self.nparams:
                continue
            n_clusters, labels = get_knn_clusters(x, max_components=x.shape[0]//2)
            if n_clusters > 1:
                new_labels = self.summaries.split(i, labels)
                all_labels[all_labels == i] = new_labels.clone()
                self.n_clusters = self.summaries.n_clusters
                self.live_points.set_labels(all_labels)

    def kill_point(self, idx=None):
        """
        Kill a point, removing it from the live points and updating the summaries
        Parameters
        ----------
        idx
            The index of the point to kill

        Returns
        -------
        sample : NSPoint
            The point that was killed
        """
        if idx is None:
            sample = self.live_points.pop()
        else:
            sample = self.live_points.pop_by_index(idx)

        # Update the summaries
        label = sample.get_labels().item()
        try:
            n_p = self.live_points.count_labels()[label]
        except IndexError:
            n_p = 0

        kill_cluster = n_p == 0

        # Add one, as we have already removed the point
        self.summaries.update(sample.get_logL(), label, n_p)
        #n_p = n_p + 1

        logweight = self.summaries.get_logXp()[label] - torch.log(torch.as_tensor(n_p + 1, device=self.device))
        sample.logweights = logweight * torch.ones_like(sample.logweights, device=self.device)
        self.dead_points.add_nspoint(sample)

        if kill_cluster:
            self.live_points.labels[self.live_points.labels > label] -= 1
            self.summaries.kill_cluster(label)
            self.n_clusters = self.summaries.n_clusters

        return sample

    def add_point(self, min_logL):
        """
        Add a new point to the live points, sampling from the prior until finding a point with higher likelihood
        than a given value
        Parameters
        ----------
        min_logL
            The minimum log-likelihood of the new point

        Returns
        -------
        """
        # Add a new sample
        newsample = self.find_new_sample(min_logL)
        newsample.logL_birth = min_logL
        self.xlogL = torch.cat((self.xlogL, min_logL + self.summaries.get_logX()))
        #print(self.xlogL)
        assert newsample.get_logL() >= min_logL, "New sample has lower likelihood than old one"

        self.live_points.add_nspoint(newsample)
        self.n_accepted += 1

    def move_one_step(self):
        """ Find highest log like, get rid of that point, and sample a new one """
        sample = self.kill_point()
        self.add_point(min_logL=sample.get_logL())

    def get_mean_logZ(self):
        """ Return the mean log evidence
        Returns
        -------
        mean_logZ : float
            The mean log evidence
        """
        return self.summaries.get_mean_logZ()

    def get_var_logZ(self):
        """ Return the variance of the log evidence
        Returns
        -------
        var_logZ : float
            The variance of the log evidence
        """
        return self.summaries.get_var_logZ()

    def terminate(self, run_time):
        """ Terminates the algorithm by adding the final live points to the dead
        points, calculating the final log evidence and acceptance rate, and printing
        a message
        Parameters
        ----------
        run_time: float
          The time taken for the algorithm to finish, in seconds
        """

        for _ in range(self.get_nlive()-1):
            self.kill_point()

        log_weights = self.dead_points.get_log_weights() + self.dead_points.get_logL() - self.summaries.get_logZ()
        self.dead_points.logweights = log_weights

        acc_rate = self.dead_points.get_size() / float(self.like_evals)

        if self.verbose:
            print('---------------------------------------------')
            print('Nested Sampling completed')
            print('Run time =', run_time, 'seconds')
            print('Acceptance rate =', acc_rate)
            print('Number of likelihood evaluations =', self.like_evals)
            print(f'logZ = {self.summaries.get_mean_logZ().item() :.4f} +/- {self.summaries.get_var_logZ().item()**0.5 :.4f}')
            print('---------------------------------------------')


    def run(self, write_to_file=False, filename=None):
        """ The main function of the algorithm. Runs the Nested sampler"""

        start_time = time.time()
        tracemalloc.start()  # Start memory profiling

        # Generate live points
        self.live_points.add_nspoint(self.sample_prior(npoints=self.nlive_ini, initial_step=True))

        # Run the algorithm
        converged = False

        # From printing and clustering updates
        prev_multiple = 0

        if write_to_file and filename is None:
            raise ValueError("Filename must be provided if write_to_file is True")

        if os.path.exists(f'{filename}_values.txt'):
            os.remove(f'{filename}_values.txt')
            os.remove(f'{filename}_logweights.txt')
            os.remove(f'{filename}_labels.txt')
            os.remove(f'{filename}_logL.txt')
            os.remove(f'{filename}_logL_birth.txt')


        nsteps = 0
        while ((self.n_clusters > 0) and (not converged)):
            self.move_one_step()
            gc.collect()

            curr_xlogL = self.xlogL[-1] - torch.max(self.xlogL)
            epsilon = self._get_epsilon()
            max_epsilon = torch.sum(epsilon) if self.clustering else epsilon
            converged = curr_xlogL < log(self.tol)

            next_multiple = (prev_multiple // self.nlive_ini + 1) * self.nlive_ini
            if self.n_accepted >= next_multiple:
                if write_to_file:
                    self.save(f'{filename}.pkl')
                    self.dead_points.write_to_file(f'{filename}')
                    self.dead_points.empty()
                    gc.collect()
                prev_multiple = next_multiple
                if self.clustering:
                    self.find_clusters()

                if self.verbose:
                    logZ_mean = self.get_mean_logZ()
                    print('---------------------------------------------')
                    print(f'logZ = {logZ_mean :.4f}, eps = {max_epsilon.item() :.4e}, {exp(curr_xlogL.item()) :.4f}')
                    if self.clustering:
                        cluster_volumes = torch.exp(self.summaries.get_logXp()).detach().numpy()
                        volume_fractions = cluster_volumes / cluster_volumes.sum()
                        logZps = self.summaries.get_logZp().detach().numpy()
                        print('---------------------------------------------')
                        for c in range(self.n_clusters):
                            if volume_fractions[c] > 1e-4:
                                print(f'Cluster {c} has volume fraction {volume_fractions[c] :.4f} and logZp = {logZps[c] :.4f}')

            nsteps += 1

        run_time = time.time() - start_time

        if write_to_file:
            self.save(f'{filename}.pkl')
            self.dead_points.write_to_file(f'{filename}')
            self.dead_points.read_from_file(f'{filename}')

        self.terminate(run_time)

        if write_to_file:
            self.dead_points.write_to_file(f'{filename}.txt')

    def convert_to_anesthetic(self):
        return anesthetic.NestedSamples(data=self.dead_points.get_values().detach().numpy(),
                                        logL=self.dead_points.get_logL().detach().numpy(),
                                        logL_birth=self.dead_points.get_logL_birth().detach().numpy(),
                                        columns=self.paramnames,
                                        labels=self.paramlabels)

    def convert_to_getdist(self):
        """ Converts the output of the algorithm to a Getdist samples object, for
        plotting the posterior.
        Returns
        -------
        getdist_samples
          A getdist samples objects with the samples
        """

        samples = self.dead_points.get_values().detach().numpy()
        weights = self.dead_points.get_weights().detach().numpy()
        loglikes = self.dead_points.get_logL().detach().numpy()

        getdist_samples = MCSamples(
            samples=samples,
            weights=weights,
            loglikes=loglikes,
            names=self.paramnames,
            labels=self.paramlabels
        )

        return getdist_samples

    def corner_plot(self, path=None):
        """
        Plot a corner plot of the samples
        :param samples: array of shape (n_samples, n_params)
        :param path: path to save the plot to
        :return:
        """
        getdist_samples = self.convert_to_getdist()
        g = plots.get_subplot_plotter()
        g.triangle_plot([getdist_samples], filled=True)
        if path is not None:
            g.export(path)
        return g
