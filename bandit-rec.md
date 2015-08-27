

ideas relevant as of 2015-08-27

* predicting the sequence 
  * extending the tour -- "next basket“ + ”previous basket" 
* recommendation with factored attributes
* TSP +  learning the objective function

----------- 
This file contains the practical aspects of bandit for recommendations -- settings for recommending trajectories and evaluations (for now). 

Discussion on 21 May, 2015 (amended 17 May)

Gaps in the current problems / approaches include 

1. predicting a sequence (e.g. playlist)
	* problem domain of location sequences seem open
   	* active setting (on discrete cases with factored attributes) seem open (? see ref below)
   	
1. next location recommendation (sub-problem of #1 above)
	*  active setting
	*  factored attributes

1. routing / TSP
	*  active setting
	*  TSP +  learning the objective function (using rec-sys)


----------- 
* reinforcement learning for playlists

Wang, X., Wang, Y., Hsu, D., & Wang, Y. (2013). Exploration in Interactive Personalized Music Recommendation: A Reinforcement Learning Approach. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 11(1), 1–22. doi:10.1145/0000000.0000000

Liebman, E., & Stone, P. (2014). DJ-MC: A Reinforcement-Learning Agent for Music Playlist Recommendation. arXiv Preprint arXiv:1401.1880. Retrieved from http://arxiv.org/abs/1401.1880, AAMAS 2015

* recommended lists (non-active, but seem relevant)

Liu, Y., Xie, M., & Lakshmanan, L. V. S. (2014). Recommending user generated item lists. Proceedings of the 8th ACM Conference on Recommender Systems - RecSys ’14, 185–192. doi:10.1145/2645710.2645750

Shuo Chen, Joshua Moore, Douglas Turnbull, Thorsten Joachims, Playlist Prediction via Metric Embedding, ACM Conference on Knowledge Discovery and Data Mining (KDD), 2012.


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

--> this evaluation doesn't seem to apply on the location sequence setting. (cannot measure `CTR`)

