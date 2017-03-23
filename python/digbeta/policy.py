"""The strategy for choosing arms"""

import random
from math import log
import scipy.stats
import kullback


class Policy:
    """Class that implements a generic index policy class."""

    def __init__(self, num_arms, amplitude=1., lower=0.):
        self.num_arms = num_arms
        self.num_draws = dict()
        self.cum_reward = dict()
        self.t = 0
        self.amplitude = amplitude
        self.lower = lower

    @property
    def name(self):
        return self.__class__.__name__

    def start_game(self):
        self.t = 1
        for arm in range(self.num_arms):
            self.num_draws[arm] = 0
            self.cum_reward[arm] = 0.0

    def get_reward(self, arm, reward):
        self.num_draws[arm] += 1
        self.cum_reward[arm] += (reward - self.lower) / self.amplitude
        self.t += 1

    def choice(self):
        """In an index policy, choose at random an arm with maximal index."""
        index = dict()
        for arm in range(self.num_arms):
            index[arm] = self.compute_index(arm)
        max_index = max(index.values())
        best_arms = [arm for arm in index.keys() if index[arm] == max_index]
        return random.choice(best_arms)

    def compute_index(self, arm):
        raise NotImplementedError


class klUCB(Policy):
    """The generic kl-UCB policy for one-parameter exponential distributions.
      """
    def __init__(self, num_arms, klucb=kullback.klucbBern):
        Policy.__init__(self, num_arms)
        self.c = 1.
        self.klucb = klucb

    def compute_index(self, arm):
        if self.num_draws[arm] == 0:
            return float('+infinity')
        else:
            # Could adapt tolerance to the value of self.t
            return self.klucb(self.cum_reward[arm] / self.num_draws[arm],
                              self.c * log(self.t) / self.num_draws[arm], 1e-4)


class UCB(klUCB):
    """The Upper Confidence Bound (UCB) index.

    The UCB policy for bounded bandits
    Reference: [Auer, Cesa-Bianchi & Fisher - Machine Learning, 2002], with constant
    set (optimally) to 1/2 rather than 2.

    Note that UCB is implemented as a special case of klUCB for the divergence
    corresponding to the Gaussian distributions, see [Garivier & Cappé - COLT, 2011].
    """
    def __init__(self, num_arms):
        klUCB.__init__(self, num_arms, lambda x, d, sig2: kullback.klucbGauss(x, d, .25))


def clopper_pearson(k, n, alpha=0.95):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    """
    lo = scipy.stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = scipy.stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


class CPUCB(Policy):
    """Clopper-Pearson UCB
    [Garivier & Cappé, COLT 2011]
    """
    def __init__(self, num_arms, c=1.01):
        """c is the parameter of the UCB"""
        Policy.__init__(self, num_arms)
        self.c = c

    def compute_index(self, arm):
        if self.num_draws[arm] == 0:
            return float('+infinity')
        else:
            lcb, ucb = clopper_pearson(self.cum_reward[arm], self.num_draws[arm],
                                       1. / (self.t**self.c))
            return ucb
