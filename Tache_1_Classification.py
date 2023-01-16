import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score



columns_= ['UserID', 'tNumerOfFollowings', 'tNumberOfFollowers', 'tNumberOfTweets',
       'tLengthOfScreenName', 'LengthOfDescriptionInUserProfile',
       'RatioFollowingFollowers', 'DureVieCompte', 'RatioTweetDureeDeVie',
       'StanderedDiviation', 'NumberOfTweetsPerDay', 'MoyenUrlTweet',
       'RatioUrlTweet', 'Ratio@Tweet', 'TimeBetweenTweet',
       'MaxTempsEntreDeuxTweet', 'Class']

legitime_polluters=pd.read_csv('output_All_users_clean.csv', sep=",", header=None , names=columns_)

# We get rid of the UserID and the Class
columns= [ 'tNumerOfFollowings', 'tNumberOfFollowers', 'tNumberOfTweets',
       'tLengthOfScreenName', 'LengthOfDescriptionInUserProfile',
       'RatioFollowingFollowers', 'DureVieCompte', 'RatioTweetDureeDeVie',
       'StanderedDiviation', 'NumberOfTweetsPerDay', 'MoyenUrlTweet',
       'RatioUrlTweet', 'Ratio@Tweet', 'TimeBetweenTweet',
       'MaxTempsEntreDeuxTweet']

X = legitime_polluters[columns] # Features
y = legitime_polluters.Class # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test

########################################DecisionTree###############################################
clf_DecisionTree = DecisionTreeClassifier()  # Create Decision Tree classifer object
# Train Decision Tree Classifer
clf_DecisionTree = clf_DecisionTree.fit(X_train,y_train)
#Predict the response for test dataset
y_pred_DecisionTree = clf_DecisionTree.predict(X_test)
# Model Accuracy, how often is the classifier correct?
acc_DecisionTree=metrics.accuracy_score(y_test, y_pred_DecisionTree)
print("Accuracy DT:", acc_DecisionTree )
### confusion_matrix
x  = confusion_matrix(y_test, y_pred_DecisionTree, labels= [1,0]) # we add labels=[1,0] to get the the values of the polluters class
print(x)
TPR_DT = x[0,0]/(x[0,0]+x[0,1])
print('true positive rate : ', TPR_DT )
FPR_DT = x[1,0]/(x[1,1]+x[1,0])
print('false positive rate : ', FPR_DT)
### F-measure
score_DecisionTree = f1_score(y_test, y_pred_DecisionTree, labels=[1,0])
print('F-Measure: %.3f' % score_DecisionTree)
### calculer la surface du courbe roc
auc_DecisionTree = roc_auc_score(y_test, y_pred_DecisionTree)
print('AUC: %.3f' % auc_DecisionTree)

# ##########################################Random Forest ####################################
#Create Random Forest Classifier
clf_RandomForest=RandomForestClassifier(n_estimators=100)
# Train Classifer
clf_RandomForest.fit(X_train,y_train)
#Predict the response for test dataset
y_pred_Random_Forest=clf_RandomForest.predict(X_test)
## calculate the accuracy
acc_RandomForest= metrics.accuracy_score(y_test, y_pred_Random_Forest)
print("Accuracy:",acc_RandomForest)
### confusion_matrix
x1= confusion_matrix(y_test, y_pred_Random_Forest, labels=[1,0])
print(x1)
TPR_RF = x1[0,0]/(x1[0,0]+x1[0,1])
print('true positive rate : ', TPR_RF )
FPR_RF = x1[1,0]/(x1[1,1]+x1[1,0])
print('false positive rate : ', FPR_RF)
### F-measure
score_RandomForest = f1_score(y_test, y_pred_Random_Forest, labels=[1,0])
print('F-Measure: %.3f' % score_RandomForest)
###3 calculate AUC
auc_RandomForest = roc_auc_score(y_test, y_pred_Random_Forest)
print('AUC: %.3f' % auc_RandomForest)

################################################## Bagging #######################################
#Create a Bagging Classifier
clf_Bagging = BaggingClassifier()
# Train Classifer
clf_Bagging.fit(X_train, y_train)
#Predict the response for test dataset
y_pred_Bagging=clf_Bagging.predict(X_test)
## calculate the accuracy
acc_Bagging = metrics.accuracy_score(y_test, y_pred_Bagging)
print("Accuracy:",acc_Bagging)
### confusion_matrix
x2= confusion_matrix(y_test, y_pred_Bagging, labels=[1,0])
print(x2)
TPR_Bagging = x2[0,0]/(x2[0,0]+x2[0,1])
print('true positive rate : ', TPR_Bagging )
FPR_Bagging = x2[1,0]/(x2[1,1]+x2[1,0])
print('false positive rate : ', FPR_Bagging)
### F-measure
score_Bagging = f1_score(y_test, y_pred_Bagging, labels=[1,0])
print('F-Measure: %.3f' % score_Bagging)
###3 calculate AUC
auc_Bagging = roc_auc_score(y_test, y_pred_Bagging)
print('AUC: %.3f' % auc_Bagging)

################################################## Boosting #######################################
#Create a Boosting Classifier
clf_Adaboost = AdaBoostClassifier(n_estimators=50,learning_rate=1)
# Train Classifer
clf_Adaboost= clf_Adaboost.fit(X_train, y_train)
#Predict the response for test dataset
y_pred_Boosting = clf_Adaboost.predict(X_test)
## calculate the accuracy
acc_Adaboost =metrics.accuracy_score(y_test, y_pred_Boosting)
print("Accuracy:",acc_Adaboost )
### confusion_matrix
x3= confusion_matrix(y_test, y_pred_Boosting, labels=[1,0])
print(x3)
TPR_Boosting = x3[0,0]/(x3[0,0]+x3[0,1])
print('true positive rate : ', TPR_Boosting )
FPR_Boosting = x3[1,0]/(x3[1,1]+x3[1,0])
print('false positive rate : ', FPR_Boosting)
### F-measure
score_Adaboost = f1_score(y_test, y_pred_Boosting, labels=[1,0])
print('F-Measure: %.3f' % score_Adaboost)
# calculate AUC
auc_Adaboost = roc_auc_score(y_test, y_pred_Boosting)
print('AUC: %.3f' % auc_Adaboost)

################################################## Gaussian #######################################
#Create a Gaussian Classifier
clf_Gaussian = GaussianNB()
# Train Classifer
clf_Gaussian= clf_Gaussian.fit(X_train, y_train)
#Predict the response for test dataset
y_pred_Gaussian = clf_Gaussian.predict(X_test)
## calculate the accuracy
acc_Gaussian = metrics.accuracy_score(y_test, y_pred_Gaussian)
print("Accuracy:",acc_Gaussian)
### confusion_matrix
x4= confusion_matrix(y_test, y_pred_Gaussian, labels=[1,0])
print(x4)
TPR_GN = x4[0,0]/(x4[0,0]+x4[0,1])
print('true positive rate : ', TPR_GN )
FPR_GN = x4[1,0]/(x4[1,1]+x4[1,0])
print('false positive rate : ', FPR_GN)
### F-measure
score_Adaboost = f1_score(y_test, y_pred_Gaussian, labels=[1,0])
print('F-Measure: %.3f' % score_Adaboost)
###3 calculate AUC
auc_Adaboost = roc_auc_score(y_test, y_pred_Gaussian)
print('AUC: %.3f' % auc_Adaboost)


















################################################## Boosting #######################################
#Create a Bagging Classifier

# Train Classifer

#Predict the response for test dataset

## calculate the accuracy

### confusion_matrix

### F-measure

###3 calculate AUC
