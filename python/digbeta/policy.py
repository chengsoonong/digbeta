"""The strategy for choosing arms"""

import random
from math import log
import kullback

class Policy:
    """Class that implements a generic index policy class."""

    def __init__(self):
        pass

    def start_game(self, num_arms):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def compute_index(self, arm):
        raise NotImplementedError

    def choice(self):
        """In an index policy, choose at random an arm with maximal index."""
        index = dict()
        for arm in range(self.num_arms):
            index[arm] = self.compute_index(arm)
        max_index = max(index.values())
        best_arms = [arm for arm in index.keys() if index[arm] == max_index]
        return random.choice(best_arms)

class klUCB(Policy):
    """The generic kl-UCB policy for one-parameter exponential distributions.
      """
    def __init__(self, num_arms, amplitude=1., lower=0., klucb=kullback.klucbBern):
        self.c = 1.
        self.num_arms = num_arms
        self.amplitude = amplitude
        self.lower = lower
        self.num_draws = dict()
        self.cum_reward = dict()
        self.klucb = klucb

    def start_game(self):
        self.t = 1
        for arm in range(self.num_arms):
            self.num_draws[arm] = 0
            self.cum_reward[arm] = 0.0

    def compute_index(self, arm):
        if self.num_draws[arm] == 0:
            return float('+infinity')
        else:
            # Could adapt tolerance to the value of self.t
            return self.klucb(self.cum_reward[arm] / self.num_draws[arm],
                              self.c * log(self.t) / self.num_draws[arm], 1e-4)

    def get_reward(self, arm, reward):
        self.num_draws[arm] += 1
        self.cum_reward[arm] += (reward - self.lower) / self.amplitude
        self.t += 1
    # Debugging code
    #print "arm " + str(arm) + " receives " + str(reward)
    #print str(self.nbDraws[arm]) + " " + str(self.cumReward[arm])

class UCB(klUCB):
    """The Upper Confidence Bound (UCB) index.

    The UCB policy for bounded bandits
    Reference: [Auer, Cesa-Bianchi & Fisher - Machine Learning, 2002], with constant
    set (optimally) to 1/2 rather than 2.
  
    Note that UCB is implemented as a special case of klUCB for the divergence 
    corresponding to the Gaussian distributions, see [Garivier & Cappé - COLT, 2011].
    """
    def __init__(self, nbArms, amplitude=1., lower=0.):
        klUCB.__init__(self, nbArms, amplitude, lower, lambda x, d, sig2: kullback.klucbGauss(x, d, .25))
