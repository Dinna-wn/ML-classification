# ML-classification
comparison of classification models

The goal of this work is to analyze and critique the performance of some classification algorithms: Decision Tree, Random Forest, Bagging, AdaBoost, Naive Bayesian Classification. 

Data:

To perform this comparison, we need to work with datasets collected from Twitter that represent two categories of users: polluting users (Content Polluters class) 
and "normal/legitimate" users (Legitimate Users class). Mainly, the data we need to analyze contains 2,353,473 tweets posted by 
22,223 malicious users (content polluters). And 3 259 693 tweets posted by 19 276 legitimate users.


Task 1: Comparative analysis using all features

Compare the performance of these algorithms based on :

      § The true positive rate (TP Rate) - of the Content Polluters class
      
      § The false positive rate (FP Rate) - of the Content Polluters class
      
      § F-measure of the Content Polluters class
      
      § The area under the ROC curve (AUC)



Task 2: comparative analysis with attribute selection

Select the 7 best features using two metrics: (1) information gain and (2) the Chi-2 (Chi Squared) test.  

Compare the performance of these algorithms based on :

      § The true positive rate (TP Rate) - of the Content Polluters class
      
      § The false positive rate (FP Rate) - of the Content Polluters class
      
      § F-measure of the Content Polluters class
      
      § The area under the ROC curve (AUC)
