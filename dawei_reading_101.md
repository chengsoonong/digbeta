Topic: Active recommendation of trajectories over time (and space)
=========== 

(1) (one of many) recommender formulation for trajectories -- predicting next location / next basket
-----------

**Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. 
Factorizing personalized Markov chains for next-basket recommendation. 
In Proceedings of the 19th international conference on World wide web (WWW '10). ACM, New York, NY, USA, 811-820.**

The authors proposed a model (FPMC) which captures both sequence effects (temporal) likes MC as well as user preferences like MF,
basically, it is a transition cube where each slice is a user-specific transition matrix of a MC.
To estimate transition probabilities, they introduced a factorization model which allows information propagation between users, items and
transitions besides deals with sparsity of data. 
The factorization approach results in less parameters which leads better generalization than full parameterized models

* insteresting: combine ideas from both successful MF and MC approaches
* estimate parameters using factorization approach is an extension of MF
* empirical study based on a reasonable sized dataset which unfortunately seems not available to others


(if you need background reading on recommender systems)

**Koren, Yehuda, Robert Bell, and Chris Volinsky. 
"Matrix factorization techniques for recommender systems." Computer 8 (2009): 30-37.**

Comprehensive introduction to the history of recommender system research,
formulating problem from simplified scenarios, capture extra information gradually by extended models, very good for beginners.


**Steffen Rendle. 2012. 
Factorization Machines with libFM. 
ACM Trans. Intell. Syst. Technol. 3, 3, Article 57 (May 2012), 22 pages. DOI=10.1145/2168752.2168771**

Explain the theory behind libFM, some practical tips of applying libFM are also provided.


(2) TSP formulations for trajectory recommentation.
-----------

**Aristides Gionis, Theodoros Lappas, Konstantinos Pelechrinis, Evimaria Terzi.
Customized tour recommendations in urban areas. 
WSDM 2014: 313-322**

The authors proposed a framework to recommend customized tours (venue's type, visiting order, budget constraint and user statisfaction).
They investigated two variants of user statisfaction, i.e. additive benefits and attraction coverage, 
proved that problems using both user satisfaction functions are NP-hard (reduce TSP to these problems) firstly 
and then designed two algorithms to solve the two recommendation problems using dynamic programming paradigm,
both the time and space complexity of the algorithms are pseudo polynomial/exponential, 
but they are fast here as the scale of the input in the experimental dataset is small.
They also designed an ($1-\epsilon$) approximation algorithm, proposed extensions of their framework, 
though without evaluating both of them in experiments.

Pros:
 * take into account both venue's type and user's visit order
 * formal proof of problem hardness
 * designed both exact and approximate algorithms
 * extend framework to incorporate factors appear in real scenarios

Cons:
 * algorithms evaluated in experiments are still exponential, though efficient in small input scales
 * dataset used in experiments is not available 
   (a part of another [dataset 2010-2011](http://infolab.tamu.edu/data/) which could be requested,
   newer and similar [datasets 2011-2012](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) are freely available)


(3) In-between -- TSP problem with a flavor of 'cutomized' preference objective. 
-----------

**Personalized Tour Recommendation based on User Interests and Points of Interest Visit Durations. 
Kwan Hui Lim, Jeffrey Chan, Christopher Leckie and Shanika Karunasekera. 
Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI'15). 
[dataset](https://sites.google.com/site/limkwanhui/datacode#ijcai15)**

The authors formulated a integer programming to solve the itinerary recommendation problem for traveller,
the approach based on an essential concept that if a user is more interested in a certain category (e.g. park, museum, etc) 
of POIs, his/her visit duration should be longer than average in general.

* novel idea to solve recommendation problem using optimisation techniques
* some issues:
  * the time the first photo taken by a user can not approximate the arrival time of that user very good in general, 
    similar problem for a user's departure time
  * users could be satisfied or not with their trips, thus, it's not wise to take the existing trips as ground truth for granted,
    a model captures users' feedback could be better.
  * users' behaviors are correlated with the availability of recommendation, i.e., people act differently given that
    recommendation service is available to them or not.


(4) Introductory reading to bandits / active learning 
-----------

(TENTATIVELY)

**Chapter 6 of ["Prediction, learning, and games"](http://www.ii.uni.wroc.pl/~lukstafi/pmwiki/uploads/AGT/Prediction_Learning_and_Games.pdf)**

**[Theory of Bandits and Online Optimisation](http://www.cs.huji.ac.il/~shais/papers/OLsurvey.pdf)**
  * The first chapter provides good introductory information of online learning
