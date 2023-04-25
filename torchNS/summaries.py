import torch
from numpy import bincount

# Default floating point type
dtype = torch.float64

class NestedSamplingSummaries:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.n_clusters = 1

        log_minus_inf = torch.log(torch.as_tensor(1e-1000, device=self.device))

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
                                               torch.log(torch.as_tensor(np + 1, device=self.device))]), 0)

        # log Zp
        self.logZp[label] = torch.logsumexp(torch.cat([self.logZ.reshape(1),
                                                       logL + self.logXp[label] -
                                                       torch.log(torch.as_tensor(np + 1, device=self.device))]), 0)

        # log Xp
        self.logXp[label] = self.logXp[label] + torch.log(torch.as_tensor(np / (np + 1), device=self.device))

        # log Z2
        self.logZ2 = torch.logsumexp(torch.cat([self.logZ2.reshape(1),
                                                    self.logZXp[label] + logL + torch.log(
                                                        torch.as_tensor(2 / (np + 1), device=self.device)),
                                                    self.logXpXq[label, label] + 2 * logL + torch.log(
                                                        torch.as_tensor(2 / (np + 1) / (np + 2), device=self.device))
                                                    ]), 0)

        # log Zp^2
        self.logZp2[label] = torch.logsumexp(torch.cat([self.logZ2.reshape(1),
                                                        self.logZXp[label] + logL + torch.log(
                                                               torch.as_tensor(2 / (np + 1), device=self.device)),
                                                        self.logXpXq[label, label] + 2 * logL + torch.log(
                                                               torch.as_tensor(2 / (np + 1) / (np + 2),
                                                                            device=self.device))
                                                        ]), 0)

        # log ZXp
        self.logZXp[label] = torch.logsumexp(
            torch.cat([self.logZXp[label].reshape(1) + torch.log(torch.as_tensor(np / (np + 1), device=self.device)),
                       self.logXpXq[label, label] + logL + torch.log(
                           torch.as_tensor(np / (np + 1) / (np + 2), device=self.device))
                       ]), 0)

        # log ZXq
        for l in range(self.n_clusters):
            if l != label:
                self.logZXp[l] = torch.logsumexp(torch.cat([self.logZXp[label].reshape(1),
                                                            self.logXpXq[label, l] + logL - torch.log(
                                                                   torch.as_tensor((np + 1), device=self.device))
                                                               ]), 0)

        # log ZpXp
        self.logZpXp[label] = torch.logsumexp(
            torch.cat([self.logZpXp[label].reshape(1) + torch.log(torch.as_tensor(np / (np + 1), device=self.device)),
                       self.logXpXq[label, label] + logL + torch.log(
                           torch.as_tensor(np / (np + 1) / (np + 2), device=self.device))
                       ]), 0)

        # log Xp^2
        self.logXpXq[label, label] = self.logXpXq[label, label] + torch.log(
            torch.as_tensor(np / (np + 2), device=self.device))

        # log XpXq
        for l in range(self.n_clusters):
            if l != label:
                self.logXpXq[label, l] = self.logXpXq[label, l] + torch.log(
                    torch.as_tensor(np / (np + 1), device=self.device))

    def split(self, cluster, new_labels):
        log_minus_inf = torch.log(torch.as_tensor(1e-1000, device=self.device))
        num_new_clusters = max(new_labels)
        n_clusters = self.n_clusters + num_new_clusters

        new_logZp = log_minus_inf.clone() * torch.ones(n_clusters, device=self.device)
        new_logXp = torch.zeros(n_clusters, device=self.device)

        new_logZp2 = log_minus_inf.clone() * torch.ones(n_clusters, device=self.device)
        new_logZXp = log_minus_inf.clone() * torch.ones(n_clusters, device=self.device)
        new_logZpXp = torch.zeros(n_clusters, device=self.device)
        new_logXpXq = torch.log(torch.eye(n_clusters, device=self.device) + 1e-1000)

        new_logZp[:self.n_clusters] = self.logZp
        new_logXp[:self.n_clusters] = self.logXp

        new_logZp2[:self.n_clusters] = self.logZp2
        new_logZXp[:self.n_clusters] = self.logZXp
        new_logZpXp[:self.n_clusters] = self.logZpXp
        new_logXpXq[:self.n_clusters, :self.n_clusters] = self.logXpXq

        n = len(new_labels)
        ns = bincount(new_labels)
        new_labels[new_labels > 0] += self.n_clusters - 1
        new_labels[new_labels == 0] = cluster
        new_idx = [0] + list(range(self.n_clusters, n_clusters))

        for idx, ni in zip(new_idx, ns):
            new_logZp[idx] = self.logZp[cluster] + torch.log(torch.as_tensor(ni / n, device=self.device))
            new_logXp[idx] = self.logXp[cluster] + torch.log(torch.as_tensor(ni / n, device=self.device))

            new_logZp2[idx] = self.logZp2[cluster] + torch.log(torch.as_tensor(ni * (ni + 1) / n / (n + 1), device=self.device))
            new_logZXp[idx] = self.logZXp[cluster] + torch.log(torch.as_tensor(ni / n, device=self.device))
            new_logZpXp[idx] = self.logZpXp[cluster] + torch.log(torch.as_tensor(ni * (ni + 1) / n / (n + 1), device=self.device))

            for l in range(n_clusters):
                if l == idx:
                    new_logXpXq[idx, l] = self.logXpXq[cluster, cluster] + torch.log(torch.as_tensor(ni * (ni + 1) / n / (n + 1), device=self.device))
                elif l in new_idx:
                    j = new_idx.index(l)
                    new_logXpXq[idx, l] = self.logXpXq[cluster, cluster] + torch.log(torch.as_tensor(ni * ns[j] / n / (n + 1), device=self.device))
                else:
                    new_logXpXq[idx, l] = self.logXpXq[cluster, l] + torch.log(torch.as_tensor(ni / n, device=self.device))

        self.n_clusters = n_clusters
        self.logZp = new_logZp
        self.logXp = new_logXp

        self.logZp2 = new_logZp2
        self.logZXp = new_logZXp
        self.logZpXp = new_logZpXp
        self.logXpXq = new_logXpXq



