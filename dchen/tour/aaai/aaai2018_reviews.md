### Assigned_Reviewer_1

#### 1. Summary

*Please summarize the main claims/contributions of the paper in your own words.*

In this paper, the authors study an important problem of trajectory recommendation (i.e., exploiting/modeling sequential structure for a sequence of point-of-interest) using improved structured SVM (SSVM). The main challenge of the studied problem is 'ensuring global cohesion' of the recommended sequence/list, which contains two specific challenges when using SSVM, i.e., 'with multiple ground truths' and 'avoiding repeated elements'. In particular, the authors design SP model, SR model, SPpath model and SRpath model for the aforementioned challenges. Empirical results on three photo trajectory datasets crawled from Flicker show the effectiveness of the proposed trajectory recommendation approaches.

#### 2. Relevance

*Is this paper relevant to an AI audience?*

Relevant to researchers in subareas only.

#### 3. Significance

*Are the results significant?*

Moderately significant.

#### 4. Novelty

*Are the problems or approaches novel?*

Novel.

#### 5. Soundness

*Is the paper technically sound?*

Technically sound.

#### 6. Evaluation

*Are claims well-supported by theoretical analysis or experimental results?*

Somewhat weak.

#### 7. Clarity

*Is the paper well-organized and clearly written?*

Good.

#### 8. Detailed Comments

*Please elaborate on your assessments and provide constructive feedback.*

My major concern about this work is about the empirical studies: 

1. the datasets are a bit small regarding the number of users in the training data
2. some existing trajectory/sequence recommendation approaches are missing

#### 9. QUESTIONS FOR THE AUTHORS

*Please provide questions for authors to address during the author feedback period.*

1. What are the time complexities of the proposed SP, SR, SPpath, SRpath models, and the baselines in the experiments? Can the proposed approaches be applied to cases with millions of users?
2. Why some existing trajectory or sequence recommendation approaches discussed in the related works not compared in the empirical studies, especially some (matrix) factorization-based approaches and neural network based approaches (e.g., RNN)?

#### 10. OVERALL SCORE

Marginally above threshold.

#### 11. CONFIDENCE

Reviewer is knowledgeable in the area.



### Assigned_Reviewer_2

#### 1. Summary

*Please summarize the main claims/contributions of the paper in your own words.*

This paper focuses on sequence recommendation, especially trajectory recommendation to recommend a list of POIs in a city to a visitor without repeats. The authors extend the structured support vector machine to deal with this problem. The SP model and SR model are proposed.


#### 2. Relevance

*Is this paper relevant to an AI audience?*

Of limited interest to an AI audience.


#### 3. Significance

*Are the results significant?*

Moderately significant.


#### 4. Novelty

*Are the problems or approaches novel?*

Somewhat novel or somewhat incremental.


#### 5. Soundness

*Is the paper technically sound?*

Has minor errors.


#### 6. Evaluation

*Are claims well-supported by theoretical analysis or experimental results?*

Somewhat weak.


#### 7. Clarity

*Is the paper well-organized and clearly written?*

Poor.


#### 8. Detailed Comments

*Please elaborate on your assessments and provide constructive feedback.*

1. The motivation of this paper is not clear. The authors argue that these features, such as global cohesion, multiple ground truths and loop elimination, are important, but they do not illustrate why previous works failed to solve these challenges. There are lots of works for trajectory recommendation, the authors didn’t analyze and compare them with the proposed method. 
2. In eliminating loops, more analysis of the SLVA algorithm is expected.
3. In Experiment, as shown in Table 3, the results of the proposed SR, SP actually do not have significant improvements. More discussion is needed. 


#### 9. QUESTIONS FOR THE AUTHORS

*Please provide questions for authors to address during the author feedback period.*

1. The motivation of this paper is not clear. The authors argue that these features, such as global cohesion, multiple ground truths and loop elimination, are important, but they do not illustrate why previous works failed to solve these challenges. There are lots of works for trajectory recommendation, the authors didn’t analyze and compare them with the proposed method. 
2. In eliminating loops, more analysis of the SLVA algorithm is expected.
3. In Experiment, as shown in Table 3, the results of the proposed SR, SP actually do not have significant improvements. More discussion is needed. 

Other minor comments:

1. The example in the section of Introduction is not clear. “while a user’s two favourite songs might be in the metal and country genres, a playlist featuring these songs in succession may be jarring.” What does this example mean?
2. The readability of the paper should be enhanced and the presentation needs to be polished for readers to better understand the paper. 

#### 10. OVERALL SCORE

Marginally above threshold.

#### 11. CONFIDENCE

Reviewer is knowledgeable in the area.



### Assigned_Reviewer_3

#### 1. Summary

*Please summarize the main claims/contributions of the paper in your own words.*

This paper study the trajectory recommendation problem by the structured SVM algorithm. The presented model can ensure the global cohesion and avoid loops in the recommended sequence. Experimental results confirms the effectiveness of the proposed model.


#### 2. Relevance

*Is this paper relevant to an AI audience?*

Relevant to researchers in subareas only.


#### 3. Significance

*Are the results significant?*

Moderately significant.


#### 4. Novelty

*Are the problems or approaches novel?*

Somewhat novel or somewhat incremental.


#### 5. Soundness

*Is the paper technically sound?*

Has minor errors.


#### 6. Evaluation

*Are claims well-supported by theoretical analysis or experimental results?*

Somewhat weak.


#### 7. Clarity

*Is the paper well-organized and clearly written?*

Satisfactory.


#### 8. Detailed Comments

*Please elaborate on your assessments and provide constructive feedback.*

In general, this paper is well written. The concerns I have mainly include the follows. 

0. The distance between two successive POIs is an important factor for trajectory recommendation. In addition, the sparseness of observed data is another issue should be considered when building the model.

1. Many details about the data set used in this paper is missing. For example, the Flickr photos only contains the GPS location and the specific POI is extracted from another API? From Table 2, It seems that each users only have about 1 to 2 trajectories in the data set and each POI will occur in more about 10 to 30 Trajectories. In practice, there will be many POIs in a city and the real data would be very sparse. 


2. Authors generate trajectories from Flickr uploaded geo-tagged photos. However, these photos may not uploaded in one same day and the some POIs may be missing. This noise and uncertainty is not considered in the presented model. In addition, the most commonly used data set for LBSN recommendation is Foursqure data set, why authors did not consider this? In addition, the data set used in the empirical study is relatively small. Authors may collect a bigger data set for the performance evaluation.


#### 9. QUESTIONS FOR THE AUTHORS

*Please provide questions for authors to address during the author feedback period.*

Please refer to the above comments which contains some questions about the experiment setups and data pre-processing.

#### 10. OVERALL SCORE

Marginally below threshold.

#### 11. CONFIDENCE

Reviewer is knowledgeable in the area.



### Meta_Reviewer_6

#### Detailed Comments

This paper study the trajectory recommendation problem by the structured SVM algorithm. Reviewers are concerned with the following aspects:
1. the motivation is not clear. some claims are not supported by any materials.
2. there are important technical details missing.
3. the data for evaluation is questionable.
