
This file contains the practical aspects of bandit for recommendations -- settings for recommending trajectories and evaluations (for now). 

Discussion on 21 May, 2015
Gaps in the current problems / approaches include

* next location recommendation
	*  active setting
	*  factored attributes

* routing / TSP
	*  active setting
	*  TSP +  learning the objective function (using rec-sys)


----------- 

* (one of many) recommender formulation for trajectories -- predicting next location 

	Chen Cheng, Haiqin Yang, Michael R. Lyu, Irwin King:
  Where You Like to Go Next: Successive Point-of-Interest Recommendation. IJCAI 2013

* TSP formulations for trajectory recommentation. 

Aristides Gionis, Theodoros Lappas, Konstantinos Pelechrinis, Evimaria Terzi:
Customized tour recommendations in urban areas. WSDM 2014: 313-322

* in-between with F100M data

Personalized Tour Recommendation based on User Interests and Points of Interest Visit Durations. 
Kwan Hui Lim, Jeffrey Chan, Christopher Leckie and Shanika Karunasekera. 
Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI'15). 
dataset: https://sites.google.com/site/limkwanhui/datacode#ijcai15


Evaluations

* Contextual bandits (Li et al)

  - Li, L., Chu, W., Langford, J., Moon, T., & Wang, X. An Unbiased Offline Evaluation of Contextual Bandit Algorithms with Generalized Linear Models. JMLR 2012
  - Li, Lihong, et al. "Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms." Proceedings of the fourth ACM international conference on Web search and data mining. ACM, 2011.
  - Li, Lihong, Wei Chu, John Langford, and Robert E. Schapire. "A contextual-bandit approach to personalized news article recommendation." In Proceedings of the 19th international conference on World wide web, pp. 661-670. ACM, 2010.

  - Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. In Advances in neural information processing systems (pp. 2249-2257).

