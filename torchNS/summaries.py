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
        self.logZpXp = log_minus_inf.clone() * torch.ones(self.n_clusters, device=self.device)
        self.logXpXq = torch.log(torch.eye(self.n_clusters, device=self.device) + 1e-1000)

        # Create tensors to store killed clusters
        # Empty 1D tensor
        self.dead_logZp = torch.empty(0, device=self.device)
        self.dead_logXp = torch.empty(0, device=self.device)

    def get_logZ(self):
        return self.logZ

    def get_logZp(self):
        return self.logZp

    def get_logXp(self):
        return self.logXp

    def get_mean_logZ(self):
        old_logZ = 2 * self.logZ - 0.5 * self.logZ2
        return old_logZ

    def get_var_logZ(self):
        old_var_logZ = self.logZ2 - 2 * self.logZ
        return old_var_logZ

    def update(self, logL, label, np):
        np = torch.as_tensor(np, dtype=torch.float64, device=self.device)
        # log Z
        self.logZ = torch.logsumexp(torch.cat([self.logZ.reshape(1),
                                               logL + self.logXp[label] -
                                               torch.log(torch.as_tensor(np + 1., device=self.device))]), 0)

        # log Zp
        self.logZp[label] = torch.logsumexp(torch.cat([self.logZp[label].reshape(1),
                                                       logL + self.logXp[label] -
                                                       torch.log(torch.as_tensor(np + 1., device=self.device))]), 0)


        # log Z2
        self.logZ2 = torch.logsumexp(torch.cat([self.logZ2.reshape(1),
                                                self.logZXp[label] + logL + torch.log(
                                                        torch.as_tensor(2. / (np + 1.), device=self.device)),
                                                self.logXpXq[label, label] + 2 * logL + torch.log(
                                                        torch.as_tensor(2. / (np + 1.) / (np + 2.), device=self.device))
                                                ]), 0)

        # log Zp^2
        self.logZp2[label] = torch.logsumexp(torch.cat([self.logZp2[label].reshape(1),
                                                        self.logZpXp[label] + logL + torch.log(
                                                               torch.as_tensor(2. / (np + 1.), device=self.device)),
                                                        self.logXpXq[label, label] + 2 * logL + torch.log(
                                                               torch.as_tensor(2. / (np + 1.) / (np + 2.),
                                                                               device=self.device))
                                                        ]), 0)


        # log ZXp
        self.logZXp[label] = torch.logsumexp(
            torch.cat([self.logZXp[label].reshape(1) + torch.log(torch.as_tensor(np / (np + 1.), device=self.device)),
                       self.logXpXq[label, label] + logL + torch.log(
                           torch.as_tensor(np / (np + 1.) / (np + 2.), device=self.device))
                       ]), 0)

        # log ZXq
        for l in range(self.n_clusters):
            if l != label:
                self.logZXp[l] = torch.logsumexp(torch.cat([self.logZXp[l].reshape(1),
                                                            self.logXpXq[label, l] + logL - torch.log(
                                                                   torch.as_tensor((np + 1.), device=self.device))
                                                               ]), 0)


        # log ZpXp
        self.logZpXp[label] = torch.logsumexp(
            torch.cat([self.logZpXp[label].reshape(1) + torch.log(torch.as_tensor(np / (np + 1.), device=self.device)),
                       self.logXpXq[label, label] + logL + torch.log(
                           torch.as_tensor(np / (np + 1.) / (np + 2.), device=self.device))
                       ]), 0)

        #assert torch.allclose(self.logZ, torch.logsumexp(self.logZp, 0)), f'{self.logZ} != {torch.logsumexp(self.logZp, 0)}'
        # if not torch.allclose(torch.logsumexp(self.logZp, 0), self.logZ):
        #     raise ValueError("logZp does not sum to logZ")

        # log Xp
        self.logXp[label] = self.logXp[label] + torch.log(torch.as_tensor(np / (np + 1.), device=self.device))

        # log Xp^2
        self.logXpXq[label, label] = self.logXpXq[label, label] + torch.log(
            torch.as_tensor(np / (np + 2.), device=self.device))

        # log XpXq
        for l in range(self.n_clusters):
            if l != label:
                self.logXpXq[label, l] = self.logXpXq[label, l] + torch.log(
                    torch.as_tensor(np / (np + 1.), device=self.device))

                self.logXpXq[l, label] = self.logXpXq[label, l].clone()


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



