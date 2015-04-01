# Related Literature

## References

Prediction, Learning, and Games
Nicolo Cesa-Bianchi, Gabor Lugosi
Cambridge University Press, 2006

Active Learning Literature Survey
Burr Settles
Computer Sciences Technical Report 1648
University of Wisconsin–Madison, 2010

Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems
Sebastien Bubeck, Nicolo Cesa-Bianchi
Foundations and Trends in Machine Learning, Vol 5, No 1, 2012, pp. 1-122

Spectral Bandits for Smooth Graph Functions
Michal Valko, Remi Munos, Branislav Kveton, Tomas Kocak
ICML 2014

Spectral Thompson Sampling
Tomas Kocak, Michal Valko, Remi Munos, Shipra Agrawal
AAAI 2014

An Analysis of Active Learning Strategies for Sequence Labeling Tasks
Burr Settles, Mark Craven
EMNLP 2008

Margin-based active learning for structured predictions
Kevin Small, Dan Roth
International Journal of Machine Learning and Cybernetics, 2010, 1:3-25

Thompson Sampling for 1-Dimensional Exponential Family Bandits
Nathaniel Korda, Emilie Kaufmann, Remi Munos
NIPS 2013

Building Bridges: Viewing Active Learning from the Multi-Armed Bandit Lens
Ravi Ganti, Alexander G. Gray
UAI 2013

From Theories to Queries: Active Learning in Practice
Burr Settles
JMLR W&CP, NIPS 2011 Workshop on Active Learning and Experimental Design

Contextual Gaussian Process Bandit Optimization.
Andreas Krause, Cheng Soon Ong
NIPS 2011

Contextual Bandit for Active Learning: Active Thompson Sampling.
Djallel Bouneffouf, Romain Laroche, Tanguy Urvoy, Raphael Feraud, Robin Allesiardo.
NIPS 2014


## Notes on UCB for graphs

Spectral Bandits for Smooth Graph Functions
Michal Valko, Remi Munos, Branislav Kveton, Tomas Kocak
ICML 2014

### Setting

Study bandit problem where the arms are the nodes of a graph and the expected payoff of pulling an arm is a smooth function on this graph.

Assume that the graph is known, and its edges represent the similarities of the nodes. At time $t$, choose a node and observe its payoff. Based on the payoff, update model.

Assume that number of nodes $N$ is large, and interested in the regime $t < N$.

