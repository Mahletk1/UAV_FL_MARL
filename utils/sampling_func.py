import numpy as np
from random import Random, shuffle

class DataPartitioner(object):
    """
    Partition a dataset into client-specific subsets (IID or Dirichlet non-IID).
    """
    def __init__(self, data, num_clients, seed=1234, NonIID='iid', alpha=0.5):
        self.data = data
        self.num_clients = num_clients

        if NonIID == 'dirichlet':
            self.partitions, self.count = self.__getDirichletData__(data, num_clients, seed, alpha)
        elif NonIID == 'iid':
            self.partitions, self.count = self.__getIIDData__(data, num_clients, seed)
        else:
            raise ValueError("NonIID must be 'iid' or 'dirichlet'")

    def use(self):
        return self.partitions, self.count

    def __getIIDData__(self, data, num_clients, seed):
        rng = Random(seed)
        data_len = len(data)
        indexes = list(range(data_len))
        rng.shuffle(indexes)

        partitions = []
        frac = 1.0 / num_clients
        for i in range(num_clients):
            part_len = int(frac * data_len)
            partitions.append(indexes[:part_len])
            indexes = indexes[part_len:]

        # class distribution (optional, for debugging)
        labelList = np.array(data.targets)
        net_cls_counts = {}
        for j in range(num_clients):
            idxs = partitions[j]
            unq, unq_cnt = np.unique(labelList[idxs], return_counts=True)
            net_cls_counts[j] = {unq[i]: unq_cnt[i] for i in range(len(unq))}

        return partitions, net_cls_counts

    def __getDirichletData__(self, data, num_clients, seed, alpha):
        np.random.seed(seed)
        K = len(np.unique(data.targets))
        labels = np.array(data.targets)
        N = len(labels)

        idx_batch = [[] for _ in range(num_clients)]

        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            splits = np.split(idx_k, proportions)
            for i in range(num_clients):
                idx_batch[i].extend(splits[i].tolist())

        # shuffle client data
        for i in range(num_clients):
            np.random.shuffle(idx_batch[i])

        # class distribution (optional)
        net_cls_counts = {}
        for j in range(num_clients):
            idxs = idx_batch[j]
            unq, unq_cnt = np.unique(labels[idxs], return_counts=True)
            net_cls_counts[j] = {unq[i]: unq_cnt[i] for i in range(len(unq))}

        return idx_batch, net_cls_counts
