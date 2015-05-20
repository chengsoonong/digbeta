"""A small test to make sure things are working"""

import numpy as np
from environment import MAB
from arm import Bernoulli
from policy import UCB, klUCB
from experiment import Evaluation

def scenario0(num_rep=10, horizon=2000):
    """Bernoulli experiment with ten arms.
    Figure 2 in Garivier & Cappe, COLT 2011"""
    probs = [0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]
    world = MAB([Bernoulli(p) for p in probs])
    policies = [UCB(world.num_arms), klUCB(world.num_arms)]

    t_save = np.linspace(100, horizon-1, 200, dtype=int)
    for pol in policies:
        w = Evaluation(world, pol, num_rep, horizon, t_save)
        print(w.mean_reward())
        print(w.mean_num_draws())
        mean_regret = w.mean_regret()
        print(mean_regret)

if __name__ == '__main__':
    scenario0()
