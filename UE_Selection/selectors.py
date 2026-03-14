import numpy as np

class BaseSelector:
    def select(self, channel_metric, K):
        raise NotImplementedError


class RandomSelector(BaseSelector):
    def select(self, channel_metric, K):
        N = len(channel_metric)
        return np.random.choice(N, K, replace=False)


class GreedyChannelSelector(BaseSelector):
    def select(self, channel_metric, K):
        # channel_metric = SNR, higher is better
        return np.argsort(channel_metric)[-K:]


class RoundRobinSelector(BaseSelector):
    def __init__(self, num_users):
        self.num_users = num_users
        self.ptr = 0

    def select(self, channel_metric, K):
        idx = (np.arange(K) + self.ptr) % self.num_users
        self.ptr = (self.ptr + K) % self.num_users
        return idx.astype(int)


class ProportionalFairSelector(BaseSelector):
    """
    PF score = instantaneous_metric / average_metric
    Here we use SNR as the metric.
    """
    def __init__(self, num_users, beta=0.95, eps=1e-6):
        self.num_users = num_users
        self.beta = beta
        self.eps = eps
        self.avg_metric = np.ones(num_users, dtype=np.float32)

    def update(self, channel_metric):
        channel_metric = np.asarray(channel_metric, dtype=np.float32)
        self.avg_metric = self.beta * self.avg_metric + (1.0 - self.beta) * channel_metric

    def select(self, channel_metric, K):
        channel_metric = np.asarray(channel_metric, dtype=np.float32)

        # shift SNR to be positive before ratio
        metric_shifted = channel_metric - np.min(channel_metric) + 1.0
        avg_shifted = self.avg_metric - np.min(self.avg_metric) + 1.0

        pf_score = metric_shifted / (avg_shifted + self.eps)
        return np.argsort(pf_score)[-K:]