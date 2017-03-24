"""Classes for representing the environment"""

import numpy as np


class Environment(object):
    """Generic class for environment, empty for the time being"""
    def __init__(self):
        pass


class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""
    def __init__(self, arms):
        self.arms = arms

    @property
    def num_arms(self):
        return len(self.arms)

    def play(self, policy, horizon):
        """Returns an object of the class Result"""
        policy.start_game()
        result = Result(self.num_arms, horizon)
        for t in range(horizon):
            choice = policy.choice()
            reward = self.arms[choice].draw()
            policy.get_reward(choice, reward)
            result.store(t, choice, reward)
        return result


class Result:
    """The Result class for analyzing the output of bandit experiments."""
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.choices = np.zeros(horizon)
        self.rewards = np.zeros(horizon)

    def store(self, t, choice, reward):
        self.choices[t] = choice
        self.rewards[t] = reward

    def get_num_pulls(self):
        if (self.num_arms == float('inf')):
            self.num_pulls = np.array([])
            pass
        else:
            num_pulls = np.zeros(self.num_arms)
            for choice in self.choices:
                num_pulls[choice] += 1
            return num_pulls

    def get_regret(self, best_expectation):
        return np.cumsum(best_expectation - self.rewards)
