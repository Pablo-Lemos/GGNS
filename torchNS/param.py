import torch

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

    def add_samples(self, values, logL, weights):
        assert all(isinstance(i, torch.Tensor)
                   for i in (values, logL, weights)), "Inputs must be tensors"
        assert values.shape[1] == self.nparams, "Wrong dimensions"
        assert values.shape[0] == weights.shape[0] == logL.shape[0], "weights and logL must be arrays"

        self.values = torch.cat([self.values, values], dim=0)
        self.logL = torch.cat([self.logL, logL], dim=0)
        self.weights = torch.cat([self.weights, weights], dim=0)

        self.currSize += values.shape[0]

    def add_nspoint(self, nspoint):
        assert isinstance(nspoint, NSPoints), "Inputs must be NSpoint"
        assert nspoint.nparams == self.nparams, "Wrong dimensions"

        self.values = torch.cat([self.values, nspoint.values], dim=0)
        self.logL = torch.cat([self.logL, nspoint.logL], dim=0)
        self.weights = torch.cat([self.weights, nspoint.weights], dim=0)
        self.currSize += nspoint.currSize

    def _sort(self):
        self.logL, indices = torch.sort(self.logL)
        self.weights = self.weights[indices]
        self.values = self.values[indices]

    def pop(self):
        self._sort()
        sample = NSPoints(self.nparams)
        sample.add_samples(values=self.values[:1],
                           weights=self.weights[:1],
                           logL=self.logL[:1])
        self.values, self.weights, self.logL = self.values[1:], \
                                               self.weights[1:], self.logL[1:]
        self.currSize -= 1
        return sample

    def get_logL(self):
        return self.logL

    def get_weights(self):
        return self.weights

    def get_values(self):
        return self.values

    def get_size(self):
        return self.currSize

