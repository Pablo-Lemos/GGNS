'''
This code creates a very Simple Nested Sampler (SNS), and applies to the examples of a unimodal and a bimodal Gaussian distribution in two dimensions. The nested sampler works, but is extremely inefficient, and to be used for pedagogic purposes. For an efficient nested sampler, I strongly recomment PolyChord (https://github.com/PolyChord/PolyChordLite)
This code was written by Pablo Lemos (UCL)
pablo.lemos.18@ucl.ac.uk
March, 2020
'''

import torch
import time
from torchNS.utils import uniform, gmm_bic, get_knn_clusters
from torchNS.param import Param, NSPoints
from torchNS.summaries import NestedSamplingSummaries
from getdist import MCSamples

# Default floating point type
dtype = torch.float64


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
        like_evals: int
          The number of likelihood evaluations
        Methods
        -------
        sample_prior(npoints, initial_step)
          Produce samples from the prior distribution
        '''

        def __init__(
                self, loglike, params, nlive=50, tol=0.1, max_nsteps=1000000, clustering=False, verbose=True, device=None):

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

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

            self.loglike = loglike
            self.params = params
            self.nlive_ini = nlive
            self.tol = tol
            self.max_nsteps = max_nsteps

            self.nparams = len(params)
            self.paramnames = []
            self.paramlabels = []
            self.dead_points = NSPoints(self.nparams)
            self.live_points = NSPoints(self.nparams)
            self.like_evals = 0
            self.verbose = verbose
            self.clustering = clustering
            self.n_clusters = 1
            self.summaries = NestedSamplingSummaries(device=self.device)
            self.cluster_volumes = torch.ones(self.n_clusters, device=self.device)

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

            # Placeholder weights (will be calculated when the point is killed)
            weights = torch.ones(len(logL), device=self.device)
            points = NSPoints(self.nparams)
            points.add_samples(values=prior_samples, weights=weights, logL=logL)

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

            Xi = torch.exp(-i / torch.tensor(self.get_nlive(), dtype=dtype, device=self.device))
            return Xi

        def get_weight(self, sample):
            ''' Calculate the weight at a given iteration, calculated as the number of
            dead points

            Returns
            -------
            weight : float
              The sample weight
            '''

            # iteration = self.dead_points.get_size()
            # X_plus = self.get_prior_volume(iteration + 2)
            # X_minus = self.get_prior_volume(iteration)
            # weight_old = 0.5 * (X_minus - X_plus)
            label = sample.get_labels().item()
            # n_p = self.live_points.count_labels()[label]
            try:
                n_p = self.live_points.count_labels()[label]
            except IndexError:
                n_p = 0
            logweight = self.summaries.get_logXp()[label] - torch.log(torch.as_tensor(n_p + 1, device=self.device))
            #logweight = self.summaries.get_logX() - torch.log(torch.as_tensor(self.live_points.get_size() + 1, device=self.device))
            weight = torch.exp(logweight)
            # print('weight', weight, 'weight_old', weight_old)
            return weight

        def get_nlive(self):
            ''' Return the number of live points
            '''
            return self.live_points.get_size()

        def _get_epsilon(self):
            ''' Find the maximum contribution to the evidence from the livepoints,
            used for the stopping criterion
            Returns
            -------
            delta_logZ : float
              The maximum contribution to the evidence from the live points
            '''

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

        def find_clusters(self):
            ''' Run a clustering algorithm to find how many clusters are present in the posterior'''
            # x = self.live_points.get_values()
            # n_clusters, labels = gmm_bic(x, max_components=self.get_nlive()//2)
            #
            # self.live_points.set_labels(labels)
            # #self.cluster_volumes = self.live_points.count_labels()/self.get_nlive() * torch.exp(self.summaries[3])
            # self.cluster_volumes = torch.exp(self.summaries[2])

            for i in range(self.n_clusters):
                all_labels = self.live_points.get_labels().detach().numpy()
                #x = self.live_points.get_values()[self.live_points.get_labels() == i]
                x = self.live_points.get_cluster(i).get_values()
                if x.shape[0] < self.nlive_ini//self.nparams:
                    continue
                #_, labels = gmm_bic(x, max_components=x.shape[0]//2)
                n_clusters, labels = get_knn_clusters(x, max_components=x.shape[0]//2)
                if n_clusters > 1:
                    new_labels = self.summaries.split(i, labels)
                    all_labels[all_labels == i] = new_labels.clone()
                    self.n_clusters = self.summaries.n_clusters
                    self.live_points.set_labels(all_labels)


        def get_cluster_live_points(self, label):
            return self.live_points.count_labels()[label]

        def kill_point(self, idx=None):
            if idx is None:
                sample = self.live_points.pop()
            else:
                #assert idx < self.live_points.get_size()
                sample = self.live_points.pop_by_index(idx)

            # if sample is None:
            #     return None

            # Update the summaries
            label = sample.get_labels().item()
            try:
                n_p = self.live_points.count_labels()[label]
                kill_cluster = False
            except IndexError:
                n_p = 0
                kill_cluster = True

            # Add one, as we have already removed the point
            n_p = n_p + 1
            self.summaries.update(sample.get_logL(), label, n_p)

            #sample.weights = self.get_weight(sample)*torch.ones_like(sample.weights, device=self.device)
            logweight = self.summaries.get_logXp()[label] - torch.log(torch.as_tensor(n_p, device=self.device))
            sample.weights = torch.exp(logweight) * torch.ones_like(sample.weights, device=self.device)
            self.dead_points.add_nspoint(sample)

            if kill_cluster:
                self.summaries.kill_cluster(label)
                self.n_clusters = self.summaries.n_clusters

            return sample

        def add_point(self, min_logL):
            # Add a new sample
            newsample = self.find_new_sample(min_logL)
            assert newsample.get_logL() > min_logL, "New sample has lower likelihood than old one"

            if self.clustering:
                # Find closest point to new sample
                values = self.live_points.get_values()
                dist = torch.sum((values - newsample.get_values())**2, dim=1)
                idx = torch.argmin(dist)

                # Assign its label to the new point
                newsample.set_labels(self.live_points.get_labels()[idx].reshape(1))
            self.live_points.add_nspoint(newsample)

        def move_one_step(self):
            ''' Find highest log like, get rid of that point, and sample a new one '''
            sample = self.kill_point()
            self.add_point(sample.get_logL())

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

        def get_mean_logZ(self):
            return self.summaries.get_mean_logZ()

        def get_var_logZ(self):
            return self.summaries.get_var_logZ()

        def terminate(self, run_time):
            ''' Terminates the algorithm by adding the final live points to the dead
            points, calculating the final log evidence and acceptance rate, and printing
            a message
            Parameters
            ----------
            run_time: float
              The time taken for the algorithm to finish, in seconds
            '''

            # weights_live = self.get_weight()
            # self.live_points.weights = weights_live*torch.ones_like(
            #     self.live_points.get_weights(), device=self.device)
            #
            # self.dead_points.add_nspoint(self.live_points)
            #
            # # Add the contribution from the live points to the evidence
            # self.add_logZ_live()
            #

            for _ in range(self.get_nlive()-1):
                self.kill_point()

            # Convert the prior weights to posterior weights
            # self.dead_points.weights *= torch.exp(
            #     self.dead_points.get_logL() - self.summaries.get_logZ())
            log_weights = torch.log(self.dead_points.get_weights()) + self.dead_points.get_logL() - self.summaries.get_logZ()
            self.dead_points.weights = torch.exp(log_weights)

            acc_rate = self.dead_points.get_size() / float(self.like_evals)

            if self.verbose:
                print('---------------------------------------------')
                print('Nested Sampling completed')
                print('Run time =', run_time, 'seconds')
                print('Acceptance rate =', acc_rate)
                print('Number of likelihood evaluations =', self.like_evals)
                print('logZ =', self.summaries.get_mean_logZ().item(), '+/-',
                      self.summaries.get_var_logZ().item()**0.5)
                print('---------------------------------------------')


        def run(self):
            ''' The main function of the algorithm. Runs the Nested sampler'''

            start_time = time.time()

            # Generate live points
            self.live_points.add_nspoint(self.sample_prior(npoints=self.nlive_ini, initial_step=True))

            # Run the algorithm
            max_epsilon = 1e1000
            nsteps = 0
            #while (self.n_clusters > 0 and self.get_nlive() > 2):
            while (self.n_clusters > 0 and max_epsilon > self.tol):
                self.move_one_step()
                # delta_logZ = self.get_delta_logZ()
                # epsilon = torch.exp(delta_logZ - self.logZ)
                epsilon = self._get_epsilon()
                max_epsilon = torch.max(epsilon) if self.clustering else epsilon
                #if self.clustering and (torch.min(epsilon) < self.tol):
                    #clusters_to_kill = torch.where(epsilon < self.tol)[0]
                    #print(f'Killing clusters {clusters_to_kill}')
                    #for c in reversed(clusters_to_kill):
                    #    self._kill_cluster(c)

                if (nsteps % self.nlive_ini == 0) and self.verbose:
                    if self.clustering:
                        self.find_clusters()

                    if self.verbose:
                        logZ_mean = self.get_mean_logZ()
                        print('---------------------------------------------')
                        print(f'logZ = {logZ_mean :.4f}, eps = {max_epsilon.item() :.4e}')
                        if self.clustering:
                            cluster_volumes = torch.exp(self.summaries.get_logXp()).detach().numpy()
                            volume_fractions = cluster_volumes / cluster_volumes.sum()
                            logZps = self.summaries.get_logZp().detach().numpy()
                            print('---------------------------------------------')
                            for c in range(self.n_clusters):
                                if volume_fractions[c] > 1e-4:
                                    print(f'Cluster {c} has volume fraction {volume_fractions[c] :.4f} and logZp = {logZps[c] :.4f}')

                nsteps += 1

            if nsteps == self.max_nsteps:
                print('WARNING: Target tolerance was not achieved after', nsteps, 'steps. Increase max_nsteps')

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


if __name__ == "__main__":
    ndims = 2
    mvn1 = torch.distributions.MultivariateNormal(loc=2 * torch.ones(ndims),
                                                  covariance_matrix=torch.diag(
                                                      0.2 * torch.ones(ndims)))

    mvn2 = torch.distributions.MultivariateNormal(loc=-1 * torch.ones(ndims),
                                                  covariance_matrix=torch.diag(
                                                      0.2 * torch.ones(ndims)))

    true_samples = torch.cat([mvn1.sample((5000,)), mvn2.sample((5000,))], dim=0)

    def get_loglike(theta):
        lp = torch.logsumexp(torch.stack([mvn1.log_prob(theta), mvn2.log_prob(theta)]), dim=-1,
                             keepdim=False) - torch.log(torch.tensor(2.0))
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

    ns = NestedSampler(
        nlive=25*ndims,
        loglike=get_loglike,
        params=params,
        clustering=True,
        verbose=True,)

    ns.run()

    # The true logZ is the inverse of the prior volume
    import numpy as np
    print('True logZ = ', np.log(1 / 10**len(params)))
    print('Number of evaluations', ns.get_like_evals())

    # import matplotlib.pyplot as plt
    # values = ns.dead_points.get_values().detach().numpy()
    # fig = plt.figure()
    # plt.scatter(values[:, 0], values[:, 1], s=ns.dead_points.get_weights().detach().numpy() * 1, alpha=0.5)
    # plt.show()

    from getdist import plots, MCSamples
    samples = ns.convert_to_getdist()
    true_samples = MCSamples(samples=true_samples.numpy(), names=[f'p{i}' for i in range(ndims)])
    g = plots.get_subplot_plotter()
    g.triangle_plot([true_samples, samples], filled=True, legend_labels=['True', 'GDNest'])
    g.export('test.png')