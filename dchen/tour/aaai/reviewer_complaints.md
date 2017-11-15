#### Experiment

1. Datasets:
   - Datasets are too small.  (ICML R3, NIPS R2, AAAI R1, AAAI R3)
   - Why not Foursquare dataset (AAAI R3)
   - Many details about the data set used is missing. For example, the Flickr photos only contains the GPS location and the specific POI is extracted from another API? Each user only have about 1 to 2 trajectories in the data set and each POI will occur in more about 10 to 30 Trajectories. In practice, there will be many POIs in a city and the real data would be very sparse. (AAAI R3)
2. Setup:
   - There is little discussion on how the feature mapping is done.  (ICML R3)
   - More details about how the training and testing sets were constructed. (ICML R3)
3. Methods:
   - Evaluation lacks mainly comparison with other methods that take the sequential structure of the data into account. (NIPS R2, NIPS R3, AAAI R1),
     - Rendle's paper (NIPS R3)
     - Matrix factorization (AAAI R1)
     - RNN (NIPS R2, AAAI R1)



#### Related work

- The existence of multiple ground truths (or multiple outputs consistent with a single ground truth) has been addressed previously in various contexts. e.g. (ICML R2) 
  - in machine translation, `Hope and Fear for Discriminative Training of Statistical Translation Models, Chiang, 2012`. 
  - in ranking problems for several combinations of performance measures and forms of supervision,  `Surrogate Functions for Maximizing Precision at the Top, Kar et al., 2015`.
- A more detailed review mentioning limitation of the other solutions. (NIPS R2)
- The authors argue that global cohesion, multiple ground truths and loop elimination are important, but they do not illustrate why previous works failed to solve these challenges.  (AAAI R2)



#### Method

- Do not fully agree on the fixed length of a recommendation sequence. Maybe a more plausible sequence with length l-1 or l+1 might exist, how can that be accounted? (NIPS R2)
- No discussion about the weaknesses of proposed technique.  (NIPS R2)
- What are the time complexities of the proposed SP, SR, SPpath, SRpath models, and the baselines in the experiments? Can the proposed approaches be applied to cases with millions of users? (AAAI R1)
- More analysis of the SLVA algorithm. (AAAI R2)
- Distance between two successive POIs is an important factor for trajectory recommendation. (AAAI R3)
- The sparseness of observed data should be considered when building the model. (AAAI R3)
- Multiple predicted trajectories have no inter-structural constraint, to what extent should they differ, is replacement of a restaurant with a museum punished? The paper claims to take care of multiple similar type entities (restaurants) in the recommendation, not clear how it achieves that. (NIPS R2)
- Authors generate trajectories from Flickr uploaded geo-tagged photos. However, these photos may not uploaded in the same day and some POIs may be missing. This noise and uncertainty is not considered in the presented model.  (AAAI R3)



#### Others

- Recommendations without personalization is more akin to a more general prediction tasks. (ICML R1)


- The training data is always bias toward the existing recommendation system and this issue will be even more severe when we consider the sequence of events because the recommendation engine react based on the user feedback. This bias can have huge impact on training and eventually on the final evaluation. Does authors have any idea how this issue can be alleviated in their approach? (NIPS R1)
- When the user makes a selection of the next item in the sequence, how can that be incorporated in the inference? (NIPS R2)

