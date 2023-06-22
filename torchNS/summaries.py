import torch
from numpy import bincount

# Default floating point type
dtype = torch.float64

class NestedSamplingSummaries:
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        log_minus_inf = torch.log(torch.as_tensor(1e-1000, device=self.device))

        self.logZ = log_minus_inf.clone()
        self.logX = torch.as_tensor(0., device=self.device)
        self.logX2 = torch.as_tensor(0., device=self.device)

        self.logZ2 = log_minus_inf.clone()
        self.logZX = log_minus_inf.clone()

        # Create tensors to store killed clusters
        # Empty 1D tensor
        self.dead_logZp = torch.empty(0, device=self.device)
        self.dead_logXp = torch.empty(0, device=self.device)

    def get_logZ(self):
        return self.logZ

    def get_logZp(self):
        return self.logZ

    def get_logXp(self):
        return self.logX.unsqueeze(0)

    def get_mean_logZ(self):
        logZ = 2 * self.logZ - 0.5 * self.logZ2
        #logZ = self.logZ
        return logZ

    def get_var_logZ(self):
        var_logZ = self.logZ2 - 2 * self.logZ
        #var_logZ = self.logZ2 - self.logZ**2
        return var_logZ

    def update(self, logL, label, np):
        np = torch.as_tensor(np, dtype=dtype, device=self.device)

        # log Z
        self.logZ = torch.logsumexp(torch.cat([self.logZ.reshape(1),
                                               logL + self.logX -
                                               torch.log(torch.as_tensor(np + 1., device=self.device))]), 0)


        # log Z2
        self.logZ2 = torch.logsumexp(torch.cat([self.logZ2.reshape(1),
                                                self.logZX + logL + torch.log(
                                                        torch.as_tensor(2. / (np + 1.), device=self.device)),
                                                self.logX2 + 2 * logL + torch.log(
                                                       torch.as_tensor(1. / (np + 1.) / (np + 2.), device=self.device))
                                                ]), 0)

        # log ZXp
        self.logZX = torch.logsumexp(
            torch.cat([self.logZX.reshape(1) + torch.log(torch.as_tensor(np / (np + 1.), device=self.device)),
                       self.logX2 + logL + torch.log(
                           torch.as_tensor(np / (np + 1.) / (np + 2.), device=self.device))
                       ]), 0)

        # log Xp
        self.logX = self.logX + torch.log(torch.as_tensor(np / (np + 1.), device=self.device))

        self.logX2 = self.logX2 + torch.log(torch.as_tensor(np / (np + 2.), device=self.device))

    def split(self, cluster, labels):
        n = len(labels)
        ns = bincount(labels)
        log_minus_inf = torch.log(torch.as_tensor(1e-1000, device=self.device))
        num_new_clusters = len(ns) - 1
        assert num_new_clusters > 0
        n_clusters = self.n_clusters + num_new_clusters

        new_logZp = log_minus_inf.clone() * torch.ones(n_clusters, device=self.device)
        new_logXp = torch.zeros(n_clusters, device=self.device)

        new_logZp2 = log_minus_inf.clone() * torch.ones(n_clusters, device=self.device)
        new_logZXp = log_minus_inf.clone() * torch.ones(n_clusters, device=self.device)
        new_logZpXp = log_minus_inf.clone() * torch.ones(n_clusters, device=self.device)
        new_logXpXq = torch.log(torch.eye(n_clusters, device=self.device) + 1e-1000)

        new_logZp[:self.n_clusters] = self.logZp
        new_logXp[:self.n_clusters] = self.logXp

        new_logZp2[:self.n_clusters] = self.logZp2
        new_logZXp[:self.n_clusters] = self.logZXp
        new_logZpXp[:self.n_clusters] = self.logZpXp
        new_logXpXq[:self.n_clusters, :self.n_clusters] = self.logXpXq

        new_labels = labels.clone()
        new_labels[labels > 0] += self.n_clusters - 1
        new_labels[labels == 0] = cluster
        new_idx = [cluster] + list(range(self.n_clusters, n_clusters))

        for idx, ni in zip(new_idx, ns):
            ni = float(ni)
            if n == 0:
                raise ValueError("Empty cluster")
            new_logZp[idx] = self.logZp[cluster] + torch.log(torch.as_tensor(ni / n, device=self.device))
            new_logXp[idx] = self.logXp[cluster] + torch.log(torch.as_tensor(ni / n, device=self.device))

            new_logZp2[idx] = self.logZp2[cluster] + torch.log(torch.as_tensor(ni * (ni + 1.) / n / (n + 1.), device=self.device))
            new_logZXp[idx] = self.logZXp[cluster] + torch.log(torch.as_tensor(ni / n, device=self.device))
            new_logZpXp[idx] = self.logZpXp[cluster] + torch.log(torch.as_tensor(ni * (ni + 1.) / n / (n + 1.), device=self.device))

            for l in range(n_clusters):
                if l == idx:
                    new_logXpXq[idx, l] = self.logXpXq[cluster, cluster] + torch.log(torch.as_tensor(ni * (ni + 1.) / n / (n + 1.), device=self.device))
                elif l in new_idx:
                    j = new_idx.index(l)
                    new_logXpXq[idx, l] = self.logXpXq[cluster, cluster] + torch.log(torch.as_tensor(ni * float(ns[j]) / n / (n + 1.), device=self.device))
                    new_logXpXq[l, idx] = new_logXpXq[idx, l].clone()
                else:
                    new_logXpXq[idx, l] = self.logXpXq[cluster, l] + torch.log(torch.as_tensor(ni / n, device=self.device))
                    new_logXpXq[l, idx] = new_logXpXq[idx, l].clone()

        self.n_clusters += num_new_clusters
        # if not torch.allclose(torch.logsumexp(new_logZp, 0), self.logZ):
        #     raise ValueError("logZp does not sum to logZ")
        self.logZp = new_logZp
        self.logXp = new_logXp

        self.logZp2 = new_logZp2
        self.logZXp = new_logZXp
        self.logZpXp = new_logZpXp
        self.logXpXq = new_logXpXq

        return new_labels

    def kill_cluster(self, idx):
        self.dead_logZp = torch.cat([self.dead_logZp, self.logZp[idx].unsqueeze(0)])
        self.dead_logXp = torch.cat([self.dead_logXp, self.logXp[idx].unsqueeze(0)])

        self.logZp = torch.cat([self.logZp[:idx], self.logZp[idx + 1:]])
        self.logXp = torch.cat([self.logXp[:idx], self.logXp[idx + 1:]])

        self.logZp2 = torch.cat([self.logZp2[:idx], self.logZp2[idx + 1:]])
        self.logZXp = torch.cat([self.logZXp[:idx], self.logZXp[idx + 1:]])
        self.logZpXp = torch.cat([self.logZpXp[:idx], self.logZpXp[idx + 1:]])

        new_logXpXq = torch.cat((self.logXpXq[:idx, :], self.logXpXq[idx + 1:, :]), dim=0)
        new_logXpXq = torch.cat((new_logXpXq[:, :idx], new_logXpXq[:, idx + 1:]), dim=1)

        self.logXpXq = new_logXpXq
        self.n_clusters -= 1


