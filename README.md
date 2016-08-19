digbeta
=======

Active learning for Big Data

## Sub-projects

* ```tour``` contains investigations on recommendations of tours, migrated to [a new repository](https://bitbucket.org/d-chen/tour-cikm16).
* ```binary_reward``` contains a summary of theoretical results about kl-UCB and KL-UCB, which is implemented in ```python```.
* ```hmm_bandit``` contains a summary of ideas for using hidden Markov models for recommendation


------


# Notes on Active Learning, Bandits, Choice and Design of Experiments (ABCDE)

There are three ideas which are often used for eliciting human
responses using machine learning predictors. At a high level they are
similar is spirit, but they have different foundations which lead to
different formulations. The ideas are active learning, bandits and
experimental design. Related to this but with literature from a different field is social choice theory, which looks at how individual preferences are aggregated.

## Overview of ABCDE

### Active Learning

Active learning considers the setting where the agent interacts with
its environment to procure a training set, rather than passively
receiving i.i.d. samples from some underlying distribution.

It is often assumed that the environment is infinite (e.g. $R^d$) and
the agent has to choose a location, $x$, to query. The oracle then returns
the label $y$. It is often assumed that there is no noise in the label,
and hence there is no benefit of querying the same point $x$ again. In
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

In contrast to active learning, experimental design considers the problem of regression, i.e. where the label $y\in R$ is a real number.

The problem to be solved in experimental design is to choose a set of
trials (say of size N) to gather enough information about the object
of interest. The goal is to maximise the information obtained about
the parameters of the model (of the object).

It is often assumed that the observations at the N trials are
independent. When N is finite this is called exact design, otherwise
it is called approximate or continuous design. The environment is
assumed to be infinite (e.g. $R^d$) and the observations are scalar real variables.


------

# Unsorted notes

* Thompson sampling
* Upper Confidence Bound

### Notes on UCB for binary rewards

In the special case when the rewards of the arms are {0,1}, we can get much tighter analysis. See [pymaBandits](http://mloss.org/software/view/415/). This is also implemented in this repository under ```python/digbeta```.


### Notes on UCB for graphs

*Spectral Bandits for Smooth Graph Functions
Michal Valko, Remi Munos, Branislav Kveton, Tomas Kocak
ICML 2014*

Study bandit problem where the arms are the nodes of a graph and the expected payoff of pulling an arm is a smooth function on this graph.

Assume that the graph is known, and its edges represent the similarities of the nodes. At time $t$, choose a node and observe its payoff. Based on the payoff, update model.

Assume that number of nodes $N$ is large, and interested in the regime $t < N$.



------

# Related Literature

This is an unsorted list of references.

* Prediction, Learning, and Games,
  Nicolo Cesa-Bianchi, Gabor Lugosi
  Cambridge University Press, 2006

* Active Learning Literature Survey
  Burr Settles
  Computer Sciences Technical Report 1648
  University of Wisconsin–Madison, 2010

* Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems
  Sebastien Bubeck, Nicolo Cesa-Bianchi
  Foundations and Trends in Machine Learning, Vol 5, No 1, 2012, pp. 1-122

* Spectral Bandits for Smooth Graph Functions
  Michal Valko, Remi Munos, Branislav Kveton, Tomas Kocak
  ICML 2014

* Spectral Thompson Sampling
  Tomas Kocak, Michal Valko, Remi Munos, Shipra Agrawal
  AAAI 2014

* An Analysis of Active Learning Strategies for Sequence Labeling Tasks
  Burr Settles, Mark Craven
  EMNLP 2008

* Margin-based active learning for structured predictions
  Kevin Small, Dan Roth
  International Journal of Machine Learning and Cybernetics, 2010, 1:3-25

* Emilie Kaufmann, Nathaniel Korda and Remi Munos
  Thompson Sampling: An Asymptotically Optimal Finite Time Analysis, ALT 2012

* Thompson Sampling for 1-Dimensional Exponential Family Bandits
  Nathaniel Korda, Emilie Kaufmann, Remi Munos
  NIPS 2013

* On Bayesian Upper Confidence Bounds for Bandit Problems
  Emilie Kaufmann, Olivier Cappe, Aurelien Garivier
  AISTATS 2012
  
* Building Bridges: Viewing Active Learning from the Multi-Armed Bandit Lens
  Ravi Ganti, Alexander G. Gray
  UAI 2013

* From Theories to Queries: Active Learning in Practice
  Burr Settles
  JMLR W&CP, NIPS 2011 Workshop on Active Learning and Experimental Design

* Contextual Gaussian Process Bandit Optimization.
  Andreas Krause, Cheng Soon Ong
  NIPS 2011

* Contextual Bandit for Active Learning: Active Thompson Sampling.
  Djallel Bouneffouf, Romain Laroche, Tanguy Urvoy, Raphael Feraud, Robin Allesiardo.
  NIPS 2014

* Towards Anytime Active Learning: Interrupting Experts to Reduce Annotation Costs
  Maria Ramirez-Loaiza, Aron Culotta, Mustafa Bilgic
  SIGKDD 2013

* Actively Learning Ontology Matching via User Interaction
  Feng Shi, Juanzi Li, Jie Tang, Guotong Xie, Hanyu Li
  ISWC 2009

* A Novel Method for Measuring Semantic Similarity for XML Schema Matching
  Buhwan Jeong, Daewon Lee, Hyunbo Cho, Jaewook Lee
  Expert Systems with Applications 2008

* Tamr Product White Paper
  http://www.tamr.com/tamr-technical-overview/

* Design of Experiments in Nonlinear Models
  Luc Pronzato, Andrej Pazman
  Springer 2013

* Optimisation in space of measures and optimal design
  Ilya Molchanov and Sergei Zuyev
  ESAIM: Probability and Statistics, Vol. 8, pp. 12-24, 2004

* Active Learning for logistic regression: an evaluation
  Andrew I. Schein and Lyle H. Ungar
  Machine Learning, 2007, 68: 235-265

* Learning to Optimize Via Information-Directed Sampling
  Daniel Russo and Benjamin Van Roy

* The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond  Aurelien Garivier and Olivier Cappe
  COLT 2011
  
* A Finite-Time Analysis of Multi-armed Bandits Problems with Kullback-Leibler Divergences
  Odalric-Ambrym Maillard, Remi Munos, Gilles Stoltz
  COLT 2011
  
* Kullback-Leibler Upper Confidence Bounds for Optimal Sequential Allocation
  Olivier Cappe, Aurelien Garivier, Odalric-Ambrym Maillard, Remi Munos, Gilles Stoltz
  Annals of Statistics, 2013
