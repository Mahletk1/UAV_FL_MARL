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

# Placeholder for future MARL
class MARLSelector(BaseSelector):
    def __init__(self, marl_agent):
        self.marl_agent = marl_agent

    def select(self, channel_metric, K):
        scores = self.marl_agent.get_scores(channel_metric)
        return np.argsort(scores)[-K:]
# -*- coding: utf-8 -*-

