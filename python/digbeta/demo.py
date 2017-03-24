"""A small test to make sure things are working"""

import numpy as np
from environment import MAB
from arm import Bernoulli
from policy import UCB, klUCB, CPUCB
from experiment import Evaluation


def scenario0(num_rep=10, horizon=2000):
    """Bernoulli experiment with ten arms.
    Figure 2 in Garivier & Cappe, COLT 2011"""
    probs = [0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]
    world = MAB([Bernoulli(p) for p in probs])
    policies = [CPUCB(world.num_arms), UCB(world.num_arms), klUCB(world.num_arms)]

    t_save = np.linspace(100, horizon - 1, 200, dtype=int)
    for pol in policies:
        print('Evaluation of %s' % pol.name)
        w = Evaluation(world, pol, num_rep, horizon, t_save)
        print('Mean reward = %1.2f' % w.mean_reward())
        print('Mean number of draws for each of %d repetitions' % num_rep)
        print(w.mean_num_draws())
        print('Mean regret')
        mean_regret = w.mean_regret()
        print(mean_regret)


if __name__ == '__main__':
    scenario0()
