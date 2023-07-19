import torch
from numpy.random import randint, choice

# Default floating point type
dtype = torch.float64

class Param():
    """
    A class to represent parameters to be sampled
    """

    def __init__(self, name, prior_type, prior, label=''):
        """
        Parameters
        ----------
        name : str
          Name of the parameter
        prior_type : str
          Type of prior used for the parameter. Must be 'Uniform' or 'Gaussian'
        prior : tuple
          A tuple with the prior values. If prior_type is Uniform, the numbers
          represent the minimum and maximum value of the prior respectively. If
          prior_type is Gaussian, they represent the mean and standard deviation
          respectively
        label : str
          LaTeX name of the parameter for plotting. Defaults to '', in which case it
          is just the name of the parameter
        """
        self.name = name
        self.prior_type = prior_type
        self.prior = prior

        if label == '':
            self.label = name
        else:
            self.label = label

        if prior_type not in ['Uniform', 'Gaussian']:
            raise ValueError("Prior type must be 'Uniform' or 'Gaussian'")

    def get_prior_type(self):
        """
        Get the prior type
        Returns
        -------
        prior_type : str
            The prior type
        """
        return self.prior_type

    def get_prior(self):
        """
        Get the prior
        Returns
        -------
        prior : tuple
            The prior
        """
        return self.prior


class NSPoints:
    """
    A class to represent a set of nested sampling points
    """
    def __init__(self, nparams, device=None):
        """
        Parameters
        ----------
        nparams : int
            Number of parameters
        device : torch.device
            Device to use. Defaults to GPU if available
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nparams = nparams
        self.values = torch.zeros([0, self.nparams], device=device)
        self.logweights = torch.zeros(size=(0,), device=device)
        self.logL = torch.ones(size=(0,), device=device)
        self.logL_birth = torch.ones(size=(0,), device=device)
        self.currSize = 0
        self.labels = torch.zeros(size=(0,), device=device, dtype=torch.int64)

    def add_samples(self, values, logL, logweights, labels=None, logL_birth=None):
        """
        Add samples to the set
        Parameters
        ----------
        values : torch.Tensor
            Tensor of shape (nsamples, nparams) with the values of the parameters
        logL : torch.Tensor
            Tensor of shape (nsamples,) with the loglikelihoods
        logweights : torch.Tensor
            Tensor of shape (nsamples,) with the logweights
        labels : torch.Tensor
            Tensor of shape (nsamples,) with the labels of the samples. Defaults to None

        Returns
        -------
        """
        assert all(isinstance(i, torch.Tensor)
                   for i in (values, logL, logweights)), "Inputs must be tensors"
        assert values.shape[1] == self.nparams, "Wrong dimensions"
        assert values.shape[0] == logweights.shape[0] == logL.shape[0], "logweights and logL must be arrays"

        self.values = torch.cat([self.values, values], dim=0)
        self.logL = torch.cat([self.logL, logL], dim=0)
        self.logweights = torch.cat([self.logweights, logweights], dim=0)
        logL_birth = torch.zeros_like(logL, device=values.device) if logL_birth is None else logL_birth
        self.logL_birth = torch.cat([self.logL_birth, logL_birth], dim=0)
        labels = torch.zeros(size=(values.shape[0],), device=values.device, dtype=torch.int64) if labels is None else labels
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.currSize += values.shape[0]

    def add_nspoint(self, nspoint):
        """
        Add a NSPoint to the set
        Parameters
        ----------
        nspoint : NSPoints
            The NSPoint to add

        Returns
        -------
        """
        assert isinstance(nspoint, NSPoints), "Inputs must be NSpoint"
        assert nspoint.nparams == self.nparams, "Wrong dimensions"

        self.values = torch.cat([self.values, nspoint.values], dim=0)
        self.logL = torch.cat([self.logL, nspoint.logL], dim=0)
        self.logL_birth = torch.cat([self.logL_birth, nspoint.logL_birth], dim=0)
        self.logweights = torch.cat([self.logweights, nspoint.logweights], dim=0)
        self.labels = torch.cat([self.labels, nspoint.labels], dim=0)
        self.currSize += nspoint.currSize

    def _sort(self):
        """
        Sort the points by loglikelihood
        Returns
        -------
        """
        self.logL, indices = torch.sort(self.logL)
        self.logL_birth = self.logL_birth[indices]
        self.logweights = self.logweights[indices]
        self.values = self.values[indices]
        self.labels = self.labels[indices]

    def pop_by_index(self, idx):
        """
        Pop a point by index
        Parameters
        ----------
        idx : int
            Index of the point to pop

        Returns
        -------
        sample : NSPoints
            The popped point

        """
        sample = NSPoints(self.nparams)
        sample.add_samples(values=self.values[idx:idx+1],
                           logweights=self.logweights[idx:idx+1],
                           logL=self.logL[idx:idx+1],
                           labels=self.labels[idx:idx+1])

        self.values = torch.cat([self.values[:idx], self.values[idx+1:]], dim=0)
        self.logweights = torch.cat([self.logweights[:idx], self.logweights[idx+1:]], dim=0)
        self.logL = torch.cat([self.logL[:idx], self.logL[idx+1:]], dim=0)
        self.logL_birth = torch.cat([self.logL_birth[:idx], self.logL_birth[idx+1:]], dim=0)
        self.labels = torch.cat([self.labels[:idx], self.labels[idx+1:]], dim=0)
        self.currSize -= 1
        return sample

    def pop(self):
        """
        Pop the point with the lowest loglikelihoo
        Returns
        -------
        sample : NSPoints
            The popped point
        """
        self._sort()
        sample = NSPoints(self.nparams)
        sample.add_samples(values=self.values[:1],
                           logweights=self.logweights[:1],
                           logL=self.logL[:1],
                           logL_birth=self.logL_birth[:1],
                           labels=self.labels[:1])
        self.values, self.logweights, self.logL, self.logL_birth, self.labels = self.values[1:], self.logweights[1:], \
                                                                                self.logL[1:], self.logL_birth[1:], \
                                                                                self.labels[1:]
        self.currSize -= 1
        return sample

    def count_labels(self):
        """
        Count the number of points for each label
        Returns
        -------
        counts : torch.Tensor
            Tensor of shape (nlabels,) with the counts
        """
        return torch.bincount(self.labels)

    def label_subset(self, label):
        """
        Get a subset of the points with a given label
        Parameters
        ----------
        label : int
            The label to select

        Returns
        -------
        sample : NSPoints
            The subset of points

        """
        idx = self.labels == label
        sample = NSPoints(self.nparams)
        sample.add_samples(values=self.values[idx],
                           logweights=self.logweights[idx],
                           logL=self.logL[idx],
                           logL_birth=self.logL_birth[idx],
                           labels=self.labels[idx])
        return sample

    def get_random_sample(self, volumes, n_samples=1):
        """
        Get a random sample of points
        Parameters
        ----------
        volumes : torch.Tensor
            Tensor of shape (npoints,) with the volumes of each cluster
        n_samples: int
            Number of samples to take

        Returns
        -------
        sample : NSPoints
            The subset of points
        """
        sample = NSPoints(self.nparams)

        # If all points have the same label
        if torch.unique(self.labels).shape[0] == 1:
            idx = randint(0, self.currSize-1, size=(n_samples,))

            sample.add_samples(values=self.values[idx],
                               logweights=self.logweights[idx],
                               logL=self.logL[idx],
                               logL_birth=self.logL_birth[idx],
                               labels=self.labels[idx])

        else:
            labels = torch.multinomial(volumes / torch.sum(volumes), num_samples=n_samples, replacement=True)
            # Calculate the number of samples to take from each label
            n_samples_per_label = torch.bincount(labels)
            sample = self.get_samples_from_labels(n_samples_per_label)
        return sample

    def get_samples_from_labels(self, n_samples_per_label):
        """
        Get a random sample of points from each label
        Parameters
        ----------
        n_samples_per_label
            Tensor of shape (nlabels,) with the number of samples to take from each label

        Returns
        -------
        sample : NSPoints
            The subset of points
        """
        sample = NSPoints(self.nparams)
        for label, n_samples in enumerate(n_samples_per_label):
            if n_samples > 0:
                subset = self.label_subset(label)
                if subset.get_size() <= 1:
                    idx = [0] * n_samples
                else:
                    assert n_samples <= subset.get_size(), "Number of samples must be less than the number of points in the subset"
                    idx = choice(subset.currSize, n_samples.item(), replace=False)

                try:
                    sample.add_samples(values=subset.values[idx],
                                       logweights=subset.logweights[idx],
                                       logL=subset.logL[idx],
                                       logL_birth=subset.logL_birth[idx],
                                       labels=subset.labels[idx])
                except IndexError:
                    raise IndexError

        return sample

    def set_labels(self, labels, idx=None):
        """
        Set the labels of the points
        Parameters
        ----------
        labels : list
        idx : list

        Returns
        -------
        """
        if idx is None:
            self.labels = torch.as_tensor(labels, device=self.values.device, dtype=torch.int64)
        else:
            assert len(labels) == len(idx), "Labels and indices must have the same length"
            self.labels[idx] = torch.as_tensor(labels, device=self.values.device, dtype=torch.int64)

    def get_cluster(self, label):
        """
        Get a subset of the points with a given label
        Parameters
        ----------
        label : int

        Returns
        -------
        sample : NSPoints
            The subset of points
        """
        idx = self.labels == label
        sample = NSPoints(self.nparams)
        sample.add_samples(values=self.values[idx],
                           logweights=self.logweights[idx],
                           logL=self.logL[idx],
                           logL_birth=self.logL_birth[idx],
                           labels=self.labels[idx])
        sample.currSize = self.logL[idx].shape[0]
        return sample

    def get_logL(self):
        self._sort()
        return self.logL

    def get_logL_birth(self):
        self._sort()
        return self.logL_birth

    def get_log_weights(self):
        self._sort()
        return self.logweights

    def get_weights(self):
        self._sort()
        return self.logweights.exp()

    def get_values(self):
        self._sort()
        return self.values

    def get_labels(self):
        self._sort()
        return self.labels

    def get_size(self):
        return self.currSize

