'''
This code creates a very Simple Nested Sampler (SNS), and applies to the examples of a unimodal and a bimodal Gaussian distribution in two dimensions. The nested sampler works, but is extremely inefficient, and to be used for pedagogic purposes. For an efficient nested sampler, I strongly recomment PolyChord (https://github.com/PolyChord/PolyChordLite)
This code was written by Pablo Lemos (UCL)
pablo.lemos.18@ucl.ac.uk
March, 2020
'''

import torch
import time
from torchNS.utils import uniform
from torchNS.param import Param, NSPoints
from getdist import MCSamples, plots

# Default floating point type
dtype = torch.float32


class NestedSampler:
    '''
    The nested sampler class
    Attributes
    ----------
    loglike: function
      The logarithm of the likelihood function
    params: ls[params]
      A list contatining all parameters, the elements belong to the parameter
      class
    nlive : int
      The number of live points. Should be set to ~25*nparams
    tol: float
      The tolerance, which decides the stopping of the algorithm
    max_nsteps: int
      The maximum number of steps to be used before stopping if the target
      tolerance is not achieved
    nparams: int
      The number of parameters
    paramnames : ls[str]
      A list containing the names of each parameter
    paramlabels : ls[str]
      A list containing the labels of each parameter
    dead_points: pd.DataFrame
      A pandas dataframe containing all the dead points
    live_points: pd.DataFrame
      A pandas dataframe containing all the live points
    logZ: float
      The logarithm of the evidence
    err_logZ: float
      The estimated error in the logZ calculation
    like_evals: int
      The number of likelihood evaluations
    Methods
    -------
    sample_prior(npoints, initial_step)
      Produce samples from the prior distribution
    '''

    def __init__(
            self, loglike, params, nlive=50, tol=0.1, max_nsteps=1000000, verbose=True, device=None):

        '''
        Parameters
        ----------
        loglike: function
          The logarithm of the likelihood function
        params: ls[params]
          A list contatining all parameters, the elements belong to the parameter
          class
        nlive : int
          The number of live points. Should be set to ~25*nparams. Defaults to 50
        tol: float
          The tolerance, which decides the stopping of the algorithm. Defaults to
          0.1
        max_nsteps: int
          The maximum number of steps to be used before stopping if the target
          tolerance is not achieved. Defaults to 10,000
        verbose: bool
            A boolean indicating if the algorithm should print information as it runs
        '''

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loglike = loglike
        self.params = params
        self.nlive = nlive
        self.tol = tol
        self.max_nsteps = max_nsteps

        self.nparams = len(params)
        self.paramnames = []
        self.paramlabels = []
        self.dead_points = NSPoints(self.nparams)
        self.live_points = NSPoints(self.nparams)
        self.logZ = -1e1000  # This is equivalent to starting with Z = 0
        self.err_logZ = -1e1000  # This is equivalent to starting with Z = 0
        self.like_evals = 0
        self.verbose = verbose

    def sample_prior(self, npoints, initial_step=False):
        ''' Produce samples from the prior distributions
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
        '''

        # Create an empty list for the samples
        prior_samples = torch.zeros([npoints, self.nparams], device=self.device)

        # Iterate over all parameters
        for i, param in enumerate(self.params):
            if initial_step == True:
                self.paramnames.append(param.name)
                self.paramlabels.append(param.label)
            if param.prior_type == 'Uniform':
                prior_samples[:,i] = uniform(low=param.prior[0],
                                            high=param.prior[1],
                                            size=npoints,
                                            dtype=dtype)

            elif param.prior_type == 'Gaussian':
                prior_samples[:,i] = torch.normal(mean=param.prior[0],
                                                  std=param.prior[1],
                                                  size=npoints,
                                                  device=self.device)

        # Calculate log likelihood
        logL = torch.zeros(npoints, device=self.device)
        for i, sample in enumerate(prior_samples):
            logL[i] = self.loglike(sample)

        weights = torch.ones(len(logL), device=self.device)
        points = NSPoints(self.nparams)
        points.add_samples(values=prior_samples, weights=weights,
                                     logL=logL)

        # Count likelihood evaluations
        self.like_evals += npoints
        return points

    def get_like_evals(self):
        return self.like_evals

    def get_prior_volume(self, i):
        ''' Calculate the prior volume for a given sample
        Parameters
        ----------
        i : int
          The current iteration

        Returns
        -------
        Xi : float
          The corresponding prior volume
        '''

        Xi = torch.exp(-i / torch.tensor(self.nlive, dtype=dtype, device=self.device))
        return Xi

    def get_weight(self):
        ''' Calculate the weight at a given iteration, calculated as the number of
        dead points

        Returns
        -------
        weight : float
          The sample weight
        '''

        iteration = self.dead_points.get_size()
        X_plus = self.get_prior_volume(iteration + 2)
        X_minus = self.get_prior_volume(iteration)

        weight = 0.5 * (X_minus - X_plus)
        return weight

    def add_logZ_live(self):
        ''' Add the contribution from the live points to the Evidence calculation
        '''

        # Find current iteration
        iteration = self.dead_points.get_size()

        # Find the weight as the prior volume
        Xm = self.get_prior_volume(iteration) * torch.ones(self.nlive, dtype=dtype, device=self.device)

        # get the evidence contribution
        logZ_live = torch.logsumexp(
            (self.live_points.get_logL() + torch.log(Xm)),0) - torch.log(
                torch.tensor(self.nlive, dtype=dtype, device=self.device))

        # Add to the current evidence
        self.logZ = torch.logsumexp(torch.tensor([self.logZ, logZ_live], device=self.device), 0)

    def get_delta_logZ(self):
        ''' Find the maximum contribution to the evidence from the livepoints,
        used for the stopping criterion
        Returns
        -------
        delta_logZ : float
          The maximum contribution to the evidence from the live points
        '''

        # Find index with minimun likelihood
        max_logL = torch.max(self.live_points.get_logL())

        # Find current iteration
        iteration = self.dead_points.get_size()

        # Get prior volume
        Xi = self.get_prior_volume(iteration)

        # Get delta_logZ as log(Xi*L)
        delta_logZ = torch.log(Xi) + max_logL

        return delta_logZ

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
            newsample = self.sample_prior(npoints=1)
            newlike = newsample.get_logL()[0]

        return newsample

    def move_one_step(self):
        ''' Find highest log like, get rid of that point, and sample a new one '''

        # # Find index with minimun likelihood
        # min_index = np.argmin(self.live_points['logL'].values)
        #
        # # Store value of likelihood at that point
        # min_like = self.live_points['logL'][min_index]
        #
        # # "Kill" that sample
        # min_sample = self.live_points.iloc[min_index]
        # min_sample['weights'] = self.get_weight()
        # self.dead_points = self.dead_points.append(
        #     min_sample, ignore_index=True, sort=True)
        # self.live_points = self.live_points.drop(min_index)
        sample = self.live_points.pop()
        sample.weights = self.get_weight()*torch.ones_like(sample.weights, device=self.device)
        self.dead_points.add_nspoint(sample)

        # Add to the log evidence
        t = torch.cat([torch.tensor([self.logZ], device=self.device),
            sample.get_logL() + torch.log(sample.get_weights())])
        self.logZ = torch.logsumexp(t, 0)

        # Add a new sample
        newsample = self.find_new_sample(sample.get_logL())
        assert newsample.get_logL() > sample.get_logL(), "New sample has lower likelihood than old one"
        self.live_points.add_nspoint(newsample)

    def _collect_priors(self):
        """
        Collect the prior ranges for each parameter
        :return: array of shape (n_params, 2)
        """
        prior_array = torch.zeros([self.nparams, 2], device=self.device)
        for i, param in enumerate(self.params):
            assert param.get_prior_type() == 'Uniform', "Only Uniform priors accepted for now"
            prior_array[i] = torch.tensor(param.get_prior(), device=self.device)

        return prior_array

    def terminate(self, run_time):
        ''' Terminates the algorithm by adding the final live points to the dead
        points, calculating the final log evidence and acceptance rate, and printing
        a message
        Parameters
        ----------
        run_time: float
          The time taken for the algorithm to finish, in seconds
        '''

        weights_live = self.get_weight()
        #TODO: change this (assigning to object attribute)
        self.live_points.weights = weights_live*torch.ones_like(
            self.live_points.get_weights(), device=self.device)

        self.dead_points.add_nspoint(self.live_points)

        # Add the contribution from the live points to the evidence
        self.add_logZ_live()

        # Convert the prior weights to posterior weights
        self.dead_points.weights *= torch.exp(
            self.dead_points.get_logL() - self.logZ)

        acc_rate = self.dead_points.get_size() / float(self.like_evals)

        if self.verbose:
            print('---------------------------------------------')
            print('Nested Sampling completed')
            print('Run time =', run_time, 'seconds')
            print('Acceptance rate =', acc_rate)
            print('Number of likelihood evaluations =', self.like_evals)
            print('logZ =', self.logZ, '+/-', self.err_logZ)
            print('---------------------------------------------')

    def run(self):
        ''' The main function of the algorithm. Runs the Nested sampler'''

        start_time = time.time()

        # Generate live points
        self.live_points.add_nspoint(self.sample_prior(npoints=self.nlive,
                                           initial_step=True))

        # Run the algorithm
        nsteps = 0
        epsilon = torch.inf
        while (nsteps < self.max_nsteps and epsilon > self.tol):
            self.move_one_step()
            delta_logZ = self.get_delta_logZ()
            epsilon = torch.exp(delta_logZ - self.logZ)

            if (nsteps % 100 == 0) and self.verbose:
                print(nsteps, 'completed, logZ =', self.logZ.item(), ', epsilon =',
                      epsilon.item())
            nsteps += 1

        if nsteps == self.max_nsteps:
            print('WARNING: Target tolerance was not achieved after', nsteps,
                  'steps. Increase max_nsteps')

        # For now I am using this simplified calculation of the error in logZ
        self.err_logZ = epsilon * abs(self.logZ)

        run_time = time.time() - start_time

        self.terminate(run_time)

    def convert_to_getdist(self):
        ''' Converts the output of the algorithm to a Getdist samples object, for
        plotting the posterior.
        Returns
        -------
        getdist_samples
          A getdist samples objects with the samples
        '''

        samples = self.dead_points.values.detach().numpy()
        weights = self.dead_points.weights.detach().numpy()
        loglikes = self.dead_points.logL.detach().numpy()

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

if __name__ == "__main__":
    ndims = 2
    mvn1 = torch.distributions.MultivariateNormal(loc=10*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.5*torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-10*torch.ones(ndims),
                                                 covariance_matrix=torch.diag(
                                                     0.5*torch.ones(ndims)))

    def get_loglike(theta):
        lp = mvn1.log_prob(theta) + mvn2.log_prob(theta)
        return lp


    p1 = Param(
        name='p1',
        prior_type='Uniform',
        prior=(-25, 25),
        label='p_1')
    p2 = Param(
        name='p2',
        prior_type='Uniform',
        prior=(-25, 25),
        label='p_2')

    params = [p1, p2]

    ns = NestedSampler(
        nlive=50,
        loglike=get_loglike,
        params=params)

    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 / 100.))
    print('Number of evaluations', ns.get_like_evals())

    from getdist import plots
    samples = ns.convert_to_getdist()
    g = plots.get_subplot_plotter()
    g.triangle_plot([samples], filled=True)
    g.export('test.png')