import torch

# Default floating point type
dtype = torch.float64

class NestedSamplingSummaries:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.n_clusters = 1

        log_minus_inf = torch.log(torch.tensor(1e-1000, device=self.device))

        self.logZ = log_minus_inf.clone()
        self.logZp = log_minus_inf.clone() * torch.ones(self.n_clusters, device=self.device)
        self.logXp = torch.zeros(self.n_clusters, device=self.device)

        self.logZ2 = log_minus_inf.clone()
        self.logZp2 = log_minus_inf.clone() * torch.ones(self.n_clusters, device=self.device)
        self.logZXp = log_minus_inf.clone() * torch.ones(self.n_clusters, device=self.device)
        self.logZpXp = torch.zeros(self.n_clusters, device=self.device)
        self.logXpXq = torch.log(torch.eye(self.n_clusters, device=self.device) + 1e-1000)

    def get_logZ(self):
        return self.logZ

    def get_logZp(self):
        return self.logZp

    def get_logXp(self):
        return self.logXp

    def get_mean_logZ(self):
        return 2 * self.logZ - 0.5 * self.logZ2

    def get_var_logZ(self):
        return self.logZ2 - 2 * self.logZ

    def update(self, logL, label, np):
        # log Z
        self.logZ = torch.logsumexp(torch.cat([self.logZ.reshape(1),
                                               logL + self.logXp[label] -
                                               torch.log(torch.tensor(np + 1, device=self.device))]), 0)

        # log Zp
        self.logZp[label] = torch.logsumexp(torch.cat([self.logZ.reshape(1),
                                                       logL + self.logXp[label] -
                                                       torch.log(torch.tensor(np + 1, device=self.device))]), 0)

        # log Xp
        self.logXp[label] = self.logXp[label] + torch.log(torch.tensor(np / (np + 1), device=self.device))

        # log Z2
        self.logZ2 = torch.logsumexp(torch.cat([self.logZ2.reshape(1),
                                                    self.logZXp[label] + logL + torch.log(
                                                        torch.tensor(2 / (np + 1), device=self.device)),
                                                    self.logXpXq[label, label] + 2 * logL + torch.log(
                                                        torch.tensor(2 / (np + 1) / (np + 2), device=self.device))
                                                    ]), 0)

        # log Zp^2
        self.logZp2[label] = torch.logsumexp(torch.cat([self.logZ2.reshape(1),
                                                        self.logZXp[label] + logL + torch.log(
                                                               torch.tensor(2 / (np + 1), device=self.device)),
                                                        self.logXpXq[label, label] + 2 * logL + torch.log(
                                                               torch.tensor(2 / (np + 1) / (np + 2),
                                                                            device=self.device))
                                                        ]), 0)

        # log ZXp
        self.logZXp[label] = torch.logsumexp(
            torch.cat([self.logZXp[label].reshape(1) + torch.log(torch.tensor(np / (np + 1), device=self.device)),
                       self.logXpXq[label, label] + logL + torch.log(
                           torch.tensor(np / (np + 1) / (np + 2), device=self.device))
                       ]), 0)

        # log ZXq
        for l in range(self.n_clusters):
            if l != label:
                self.logZXp[l] = torch.logsumexp(torch.cat([self.logZXp[label].reshape(1),
                                                            self.logXpXq[label, l] + logL - torch.log(
                                                                   torch.tensor((np + 1), device=self.device))
                                                               ]), 0)

        # log ZpXp
        self.logZpXp[label] = torch.logsumexp(
            torch.cat([self.logZpXp[label].reshape(1) + torch.log(torch.tensor(np / (np + 1), device=self.device)),
                       self.logXpXq[label, label] + logL + torch.log(
                           torch.tensor(np / (np + 1) / (np + 2), device=self.device))
                       ]), 0)

        # log Xp^2
        self.logXpXq[label, label] = self.logXpXq[label, label] + torch.log(
            torch.tensor(np / (np + 2), device=self.device))

        # log XpXq
        for l in range(self.n_clusters):
            if l != label:
                self.logXpXq[label, l] = self.logXpXq[label, l] + torch.log(
                    torch.tensor(np / (np + 1), device=self.device))
