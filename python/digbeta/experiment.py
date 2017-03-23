"""Classes for conducting experiments with Bandits"""

import numpy as np


class Evaluation:
    """
    Evaluating the performance of a policy in multi-armed bandit problems.
    """
    def __init__(self, env, pol, num_repetitions, horizon, tsav=[]):
        if len(tsav) > 0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon, dtype=int)
        self.env = env
        self.pol = pol
        self.num_repetitions = num_repetitions
        self.horizon = horizon
        self.num_arms = env.num_arms
        self.num_pulls = np.zeros((self.num_repetitions, self.num_arms))
        self.cum_reward = np.zeros((self.num_repetitions, len(self.tsav)))

        for k in range(num_repetitions):
            if num_repetitions < 10 or k % (num_repetitions / 10) == 0:
                print('Repetition %d of %d' % (k + 1, num_repetitions))
            result = env.play(pol, horizon)
            self.num_pulls[k, :] = result.get_num_pulls()
            self.cum_reward[k, :] = np.cumsum(result.rewards)[self.tsav]

    def mean_reward(self):
        return sum(self.cum_reward[:, -1]) / len(self.cum_reward[:, -1])

    def mean_num_draws(self):
        return np.mean(self.num_pulls, 0)

    def mean_regret(self):
        return ((1 + self.tsav) * max([arm.expectation for arm in self.env.arms]) -
                np.mean(self.cum_reward, 0))
