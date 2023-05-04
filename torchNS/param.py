import torch
from numpy.random import randint

# Default floating point type
dtype = torch.float32


class Param():
    '''
    A class to represent parameters to be sampled
    Attributes
    ----------
    name : str
      Name of the parameter
    label : str
      LaTeX name of the parameter for plotting
    prior_type: str
      Type of prior used for the parameter. Must be 'Uniform' or 'Gaussian'
    prior: tuple
      A tuple with the prior values. If prior_type is Uniform, the numbers
      represent the minimum and maximum value of the prior respectively. If
      prior_type is Gaussian, they represent the mean and standard deviation
      respectively
    '''

    def __init__(self, name, prior_type, prior, label=''):
        '''
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
        '''

        self.name = name
        self.prior_type = prior_type
        self.prior = prior

        if label == '':
            self.label = name
        else:
            self.label = label

        if (prior_type not in ['Uniform', 'Gaussian']):
            print(
                "ERROR, prior type unknown. Only 'Uniform' or 'Gaussian' can be used")
            sys.exit()

    def get_prior_type(self):
        return self.prior_type

    def get_prior(self):
        return self.prior


class NSPoints:
    def __init__(self, nparams, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nparams = nparams
        self.values = torch.zeros([0, self.nparams], device=device)
        self.weights = torch.ones(size=(0,), device=device)
        self.logL = torch.ones(size=(0,), device=device)
        self.currSize = 0
        self.labels = torch.zeros(size=(0,), device=device, dtype=torch.int64)

    def add_samples(self, values, logL, weights, labels=None):
        assert all(isinstance(i, torch.Tensor)
                   for i in (values, logL, weights)), "Inputs must be tensors"
        assert values.shape[1] == self.nparams, "Wrong dimensions"
        assert values.shape[0] == weights.shape[0] == logL.shape[0], "weights and logL must be arrays"

        self.values = torch.cat([self.values, values], dim=0)
        self.logL = torch.cat([self.logL, logL], dim=0)
        self.weights = torch.cat([self.weights, weights], dim=0)
        labels = torch.zeros(size=(values.shape[0],), device=values.device, dtype=torch.int64) if labels is None else labels
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.currSize += values.shape[0]

    def add_nspoint(self, nspoint):
        assert isinstance(nspoint, NSPoints), "Inputs must be NSpoint"
        assert nspoint.nparams == self.nparams, "Wrong dimensions"

        self.values = torch.cat([self.values, nspoint.values], dim=0)
        self.logL = torch.cat([self.logL, nspoint.logL], dim=0)
        self.weights = torch.cat([self.weights, nspoint.weights], dim=0)
        self.labels = torch.cat([self.labels, nspoint.labels], dim=0)
        self.currSize += nspoint.currSize

    def _sort(self):
        self.logL, indices = torch.sort(self.logL)
        self.weights = self.weights[indices]
        self.values = self.values[indices]
        self.labels = self.labels[indices]

    def pop_by_index(self, idx):
        sample = NSPoints(self.nparams)
        sample.add_samples(values=self.values[idx:idx+1],
                           weights=self.weights[idx:idx+1],
                           logL=self.logL[idx:idx+1],
                           labels=self.labels[idx:idx+1])

        self.values = torch.cat([self.values[:idx], self.values[idx+1:]], dim=0)
        self.weights = torch.cat([self.weights[:idx], self.weights[idx+1:]], dim=0)
        self.logL = torch.cat([self.logL[:idx], self.logL[idx+1:]], dim=0)
        self.labels = torch.cat([self.labels[:idx], self.labels[idx+1:]], dim=0)
        self.currSize -= 1
        return sample

    def pop(self):
        self._sort()
        sample = NSPoints(self.nparams)
        sample.add_samples(values=self.values[:1],
                           weights=self.weights[:1],
                           logL=self.logL[:1],
                           labels=self.labels[:1])
        self.values, self.weights, self.logL, self.labels = self.values[1:], self.weights[1:], self.logL[1:], \
                                                            self.labels[1:]
        self.currSize -= 1
        return sample

    def count_labels(self):
        return torch.bincount(self.labels)

    def label_subset(self, label):
        idx = self.labels == label
        sample = NSPoints(self.nparams)
        sample.add_samples(values=self.values[idx],
                           weights=self.weights[idx],
                           logL=self.logL[idx],
                           labels=self.labels[idx])
        return sample

    def get_random_sample(self, volumes, n_samples=1):
        sample = NSPoints(self.nparams)

        # If all points have the same label
        if torch.unique(self.labels).shape[0] == 1:
            idx = randint(0, self.currSize-1, size=(n_samples,))

            sample.add_samples(values=self.values[idx],
                               weights=self.weights[idx],
                               logL=self.logL[idx],
                               labels=self.labels[idx])

        else:
            labels = torch.multinomial(volumes / torch.sum(volumes), num_samples=n_samples, replacement=True)
            #subset = self.label_subset(label)
            # while subset.get_size() < 1:
            #     label = torch.multinomial(volumes / torch.sum(volumes), n_samples, replacement=True)
            #     subset = self.label_subset(label)

            # Calculate the number of samples to take from each label
            n_samples_per_label = torch.bincount(labels)#, minlength=torch.max(labels)+1)
            for label, n_samples in enumerate(n_samples_per_label):
                if n_samples > 0:
                    subset = self.label_subset(label)
                    if subset.get_size() <= 1:
                        idx = [0]*n_samples
                    else:
                        idx = randint(0, subset.currSize-1, size=(n_samples,))

                    sample.add_samples(values=subset.values[idx],
                                       weights=subset.weights[idx],
                                       logL=subset.logL[idx],
                                       labels=subset.labels[idx])
        return sample

    def set_labels(self, labels, idx=None):
        if idx is None:
            self.labels = torch.as_tensor(labels, device=self.values.device, dtype=torch.int64)
        else:
            assert len(labels) == len(idx), "Labels and indices must have the same length"
            self.labels[idx] = torch.as_tensor(labels, device=self.values.device, dtype=torch.int64)

    def get_cluster(self, label):
        idx = self.labels == label
        sample = NSPoints(self.nparams)
        sample.add_samples(values=self.values[idx],
                           weights=self.weights[idx],
                           logL=self.logL[idx],
                           labels=self.labels[idx])
        sample.currSize = self.logL[idx].shape[0]
        return sample

    def get_logL(self):
        self._sort()
        return self.logL

    def get_weights(self):
        self._sort()
        return self.weights

    def get_values(self):
        self._sort()
        return self.values

    def get_labels(self):
        self._sort()
        return self.labels

    def get_size(self):
        return self.currSize

