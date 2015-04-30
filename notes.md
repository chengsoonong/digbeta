# Notes on Active Learning, Bandits and Experimental Design

There are three ideas which are often used for eliciting human
responses using machine learning predictors. At a high level they are
similar is spirit, but they have different foundations which lead to
different formulations. The ideas are active learning, bandits and
experimental design (ABED).

## Overview of ABED

### Active Learning

Active learning considers the setting where the agent interacts with
its environment to procure a training set, rather than passively
receiving i.i.d. samples from some underlying distribution.

It is often assumed that the environment is infinite (e.g. R^d) and
the agent has to choose a location, x, to query. The oracle then returns
the label y. It is often assumed that there is no noise in the label,
and hence there is no benefit of querying the same point x again. In
many practical applications, the environment is considered to be
finite (but large). This is called the pool-based active learning.

The active learning algorithm is often compared to the passive
learning algorithm.

### Bandits

A bandit problem is a sequential allocation problem defined by a set
of actions. The agent chooses an action at each time step, and the
environment returns a reward. The aim of the agent is to maximise reward.

In basic settings, the set of actions is considered to be
finite. There are three fundamental formalisations of the bandit
problem, depending on the assumed nature of the reward process:
stochastic, adversarial and Markovian. In all three settings the
reward is uncertain, and hence the agent may have to play a particular
action repeatedly.

The agent is compared to a static agent which has played the best
action. This difference in reward is called regret.

### Experimental Design

The problem to be solved in experimental design is to choose a set of
trials (say of size N) to gather enough information about the object
of interest. The goal is to maximise the information obtained about
the parameters of the model (of the object).

It is often assumed that the observations at the N trials are
independent. When N is finite this is called exact design, otherwise
it is called approximate or continuous design. The environment is
assumed to be infinite (e.g. R^d) and the observations are scalar real
variables.






# Active Learning

# Bandits

## Thompson sampling

## Upper Confidence Bound

### Notes on UCB for binary rewards

In the special case when the rewards of the arms are {0,1}, we can get much tighter analysis. See [pymaBandits](http://mloss.org/software/view/415/).


### Notes on UCB for graphs

*Spectral Bandits for Smooth Graph Functions
Michal Valko, Remi Munos, Branislav Kveton, Tomas Kocak
ICML 2014*

Study bandit problem where the arms are the nodes of a graph and the expected payoff of pulling an arm is a smooth function on this graph.

Assume that the graph is known, and its edges represent the similarities of the nodes. At time $t$, choose a node and observe its payoff. Based on the payoff, update model.

Assume that number of nodes $N$ is large, and interested in the regime $t < N$.

# Experimental Design
