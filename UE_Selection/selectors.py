import numpy as np

class BaseSelector:
    def select(self, channel_metric, K):
        raise NotImplementedError

class RandomSelector(BaseSelector):
    def select(self, channel_metric, K):
        N = len(channel_metric) #add seed here
        # print('this is the channel metric', channel_metric)
        return np.random.choice(N, K, replace=False)

class GreedyChannelSelector(BaseSelector):
    def select(self, channel_metric, K):
        # now channel_metric is SNR (higher is better)
        return np.argsort(channel_metric)[-K:]



