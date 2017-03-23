"""Model the distributions available when pulling an arm."""

from random import random


class Arm:
    """Abstract base class"""
    def __init__(self):
        pass

    def draw(self):
        raise NotImplementedError


class Bernoulli(Arm):
    """Bernoulli distributed arm."""
    def __init__(self, p):
        self.p = p
        self.expectation = p

    def draw(self):
        return float(random() < self.p)
