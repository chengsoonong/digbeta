"""Computation of Kullback-Leibler divergence"""

from math import log, sqrt, exp
import numpy as np

eps = 1e-15

def klBern(x, y):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1-eps)
    y = min(max(y, eps), 1-eps)
    return x*log(x/y) + (1-x)*log((1-x)/(1-y))

def klPoisson(x, y):
    """Kullback-Leibler divergence for Poisson distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return y-x+x*log(x/y)



def klucb(x, d, div, upperbound, lowerbound=-float('inf'), precision=1e-6):
    """The generic klUCB index computation.

    Input args.: x, d, div, upperbound, lowerbound=-float('inf'), precision=1e-6
    where div is the KL divergence to be used.
    """
    l = max(x, lowerbound)
    u = upperbound
    while u-l>precision:
        m = (l+u)/2
        if div(x, m)>d:
            u = m
        else:
            l = m
    return (l+u)/2

def klucbGauss(x, d, sig2=1., precision=0.):
    """klUCB index computation for Gaussian distributions.

    Note that it does not require any search.
    """
    return x + sqrt(2*sig2*d)

def klucbPoisson(x, d, precision=1e-6):
    """klUCB index computation for Poisson distributions."""
    upperbound = x+d+sqrt(d*d+2*x*d) # looks safe, to check: left (Gaussian) tail of Poisson dev
    return klucb(x, d, klPoisson, upperbound, precision)

def klucbBern(x, d, precision=1e-6):
    """klUCB index computation for Bernoulli distributions."""
    upperbound = min(1., klucbGauss(x, d))
    #upperbound = min(1.,klucbPoisson(x,d)) # also safe, and better ?
    return klucb(x, d, klBern, upperbound, precision)

