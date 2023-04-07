# DS_Applied_ML_competition_2022
University of Southern Denmark 2022, Applied Machine Learning Classroom Kaggle Competition, anonymized data (so you cannot cheat). 

Dataset consists of training set 48000 observations, 204 features. Test set is for predictions for kaggle leaderboard (invite only - i.e. only 3rd generation Data Science students at SDU, taking the applied machine learning course by professor Christian Møller Dahl could submit solutions). Of approx 100 students, some 30 people (or groups) submitted solutions.

First place was dominating for months, approximately 90 percent performance. I tried applying different feature selection methods, dimensionality reduction methods, and tried these on about 15 different models. They were close, but they did not beat the leader. 
In addition, what I noticed that it was very clear, that our models performed much better precision and recall on class 0 and 2 vs class 1. Here is two random examples from the many models i ran: 

AdaBoost Tuned:
| Class | Precision | Recall | Support |
|-------|-----------|--------|---------|
| 0     | 0.92      | 0.55   | 2396    |
| 1     | 0.65      | 0.92   | 3310    |
| 2     | 0.92      | 0.83   | 3894    |
|-------|-----------|--------|---------|
| Macro Avg      | 0.83      | 0.77   |    9600     |
| Weighted Avg       | 0.83          | 0.79 |      9600   |
|-------|---------------|------|---------|

XGBoost Tuned:
| Class | Precision | Recall | Support |
|-------|-----------|--------|---------|
| 0     | 0.86      | 0.81   | 2396    |
| 1     | 0.78      | 0.80   | 3310    |
| 2     | 0.87      | 0.89   | 3894    |
|-------|-----------|--------|---------|
| Macro Avg      | 0.84      | 0.83   |    9600     |
| Weighted Avg       | 0.84          | 0.84 |      9600   |
|-------|---------------|------|---------|

Months went by, and then it hit me. If Class 1 is difficult to distinguish from class 0 and 2, maybe a classifier would have an easier job if it only would have to separate either class 0 from 1 or class 2 from 1. 

So, I chose to separate the training set into three subsets X_train_0_1, X_train_1_2 and X_train_0_2. The idea then would be to train a model on each subset, use the subsets to predict on the FULL validation and test set, and then average those predictions (ensemble). 

As my notebook will show, it worked, and it worked so well that I achieved 1st place. 

My approach can be summarized as:

1. Load data, split data into train, test, validation and normalize the data.
2. Apply Boruta Feature selection algorithm (my favorite feature selection algorithm)
3. Split train set into three subsets: X_train_0_1, X_train_1_2 and X_train_0_2
4. Tune a model on each subset
5. Predict on full validation and testset
6. Average predictions
7. Evaluate prediction


My solution: Ensemble of CatBoost Tuned on the three subsets:
| Class | Precision | Recall | Support |
|-------|-----------|--------|---------|
| 0     | 0.93      | 0.87   | 2462    |
| 1     | 0.89      | 0.91   | 3325    |
| 2     | 0.92      | 0.94   | 3813    |
|-------|-----------|--------|---------|
| Macro Avg      | 0.91      | 0.91   |    9600     |
| Weighted Avg       | 0.91          | 0.91 |      9600   |
|-------|---------------|------|---------|


For ppl interested in Feature selection, Boruta, I describe it here:

The Boruta Feature Selection Algorithm

As given by the paper:
1.	Extend the information system by adding copies of all variables (the information system is always extended by at least 5 shadow attributes, even if the number of attributes in the original set is lower than 5).
2.	Shuffle the added attributes to remove their correlations with the response.
3.	Run a random forest classifier on the extended information system and gather the Z scores computed.
4.	Find the maximum Z score among shadow attributes (MZSA), and then assign a hit to every attribute that scored better than MZSA.
5.	For each attribute with undetermined importance perform a two-sided test of equality with the MZSA.
6.	Deem the attributes significantly lower than MZSA as ‘unimportant’ and permanently remove them from the information system.
7.	Deem the attributes which have importance significantly higher than MZSA as ‘important’.
8.	Remove all shadow attributes.
9.	Repeat the procedure until the importance is assigned for all the attributes, or the algorithm has reached the previously set limit of the random forest runs.

Because shadow features (MZSA) are useless, Boruta basically states that no feature should have worse feature importance scores than shadow features. It cant be important to the classification or regression problem.  

The shuffling has two functionalities, it breaks down multicollinearity and causes shadow features to have the same marginal distributions as original features, but they are independent of the target both unconditionally and conditional on original features.

In essence, The Boruta algorithm treats the test as a Bernoulli trial. That is, Boruta applied a statistical hypothesis test to support feature selection. Given that shadow features are randomly generated, we have a randomized control experiment with two outcomes, 'higher feature importance score than its shadow feature' 'not higher', that we can repeat k independent times, where we count the number of hits hi of each feature Xi.

An original feature is considered to have received a hit when its feature importance score is higher than the scores of all shadow features.

There are three hypotheses:


The Null Hypothesis H0

We do not know apriori whether feature xi is useful or not. We expect the outcomes of the test to follow a Binomial distribution with rameters k and p before running the test by saying that, in each of the k runs, there is a 50% chance that feature xi will receive a hit (i.e. a probability p=0.5 p=0.5). 

The CDF of a Binomial distribution allows us to compute the two-sided symmetric confidence intervals [mq(k), Mq(k)].


The Alternate Hypothesis H1

When the number of hits hi observed after k runs exceeds Mq(k) we reject the hypothesis H0, that is we believe that feature Xi is more likely to be useful than not.

We have a second alternative hypothessis, that represents the fact that even after say K=100 iterations, results are inconclusive.


The Alternate Hypothesis H2

Feature xi is useless When the number of hits hi observed after k runs is lower than mq(k), we reject H0 with the outcome that we lean towards Xi being more useless than useful. 

This happens occasionally, and you can either choose to change thresholds on Boruta (alpha or the stringency=perc) or you can extend >K to more trials. Eventually it should converge on a decision.

Feature importances are fundamentally flawed. Feature importance score implicitly state that the model is competent. There are high risks that if model is overfitted, you would base your feature selection decisions on wrong premises. What makes Boruta Feature selection more robust is the fact that we first of all test each feature against a randomized copy, and the fact that we rerun the modelling, say 100 times. The chance of overfitting to any single feature, becomes much smaller with a large K. 
