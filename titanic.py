# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:21:26 2021

@author: jason chan

CS412 FINAL PROJECT
"""

# import the modules needed
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

# load train set
train_df = pd.read_csv('train.csv')

# show the dimensions of train set
print(train_df.shape)

# show the first 5 rows in train set
print(train_df.head(5))

# exploratory data analysis
print(train_df.info())
print(train_df.describe())

# load test set
test_df = pd.read_csv('test.csv')

# show the dimensions of test set
print(test_df.shape)

# boolean: embarkment
for k in train_df.Embarked.unique():
    if type(k) == str:
        train_df['emb_' + k] = (train_df.Embarked == k) * 1

# boolean: is_male
train_df['is_male'] = (train_df.Sex == 'male') * 1

# boolean: has_cabin
train_df.loc[:, 'has_cabin'] = 0
train_df.loc[train_df.Cabin.isna(), 'has_cabin'] = 1

# fill in the missing values for age
train_df.loc[train_df.Age.isna(), 'Age'] = 100


# model preprocessing
#features = train_df[["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]]
features = train_df[["Pclass", "Age", "Sex", "SibSp", "Parch", "Embarked"]]
Y = train_df["Survived"]

# standardisation of the data
#sc = StandardScaler()
#X = sc.fit_transform(X)
#X_TEST = sc.transform(X_TEST)

# one hot encoding
one_hot_encoded_training_predictors = pd.get_dummies(features)
one_hot_encoded_training_predictors.head()
X = one_hot_encoded_training_predictors

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1)


# Logistic Regression Algorithm
LR = LogisticRegression(max_iter = 1000) 
#solver = 'lbfgs' , solver = liblinear
LR.fit(X_train, y_train)

# predicting the values for LR
y_pred_LR = LR.predict(X_test)

# show train/test split results for LR algorithm
print('Train/Test split results:')
print(LR.__class__.__name__+" accuracy is %2.5f" % accuracy_score(y_test, y_pred_LR))
#print(LR.__class__.__name__+" accuracy is: {:}" .format(LR.score(X_test, y_test)))

# using cross validation for LR algorithm
scores_accuracy = cross_val_score(LR, X, Y, cv = 10, scoring = 'accuracy')

# show cross validation results of LR algorithm
print('Cross Validation results:')
print(LR.__class__.__name__+" accuracy is %2.5f" % scores_accuracy.mean())

# using ensemble method (bagging) for LR algorithm
bag_LR = BaggingClassifier(
    base_estimator = LR, 
    n_estimators = 10, 
    n_jobs = -1, 
    random_state = 0
    )

bag_LR.fit(X_train, y_train)

# show ensemble method (bagging) results of LR algorithm
print('Bagging results:')
#print(bag_LR.score(X_test, y_test))
print(LR.__class__.__name__+" accuracy is %2.5f" % bag_LR.score(X_test, y_test))

# show classification report of LR algorithm
print(classification_report(y_test, y_pred_LR))

# visualization: confusion matrix of LR
print('Confusion Matrix:')
visualize_LR = plot_confusion_matrix(LR, X_test, y_test, cmap = plt.cm.Reds, values_format = 'd')


# Decision Tree Classifier
DTC = DecisionTreeClassifier(max_depth = 5)
DTC.fit(X_train, y_train)

# predicting the values for DTC
y_pred_DTC = DTC.predict(X_test)

# show train/test split results of DTC algorithm
print('Train/Test split results:')
print(DTC.__class__.__name__+" accuracy is %2.5f" % accuracy_score(y_test, y_pred_DTC))

# using cross validation for DTC algorithm
scores_accuracy = cross_val_score(DTC, X, Y, cv = 10, scoring = 'accuracy')

# show cross validation results of DTC algorithm
print('Cross Validation results:')
print(DTC.__class__.__name__ + " accuracy is %2.5f" % scores_accuracy.mean())

# using ensemble method (bagging) for DTC algorithm
bag_DTC = BaggingClassifier(
    base_estimator = DTC, 
    n_estimators = 10, 
    n_jobs = -1, 
    random_state = 0
    )

bag_DTC.fit(X_train, y_train)

# show ensemble method (bagging) results of DTC algorithm
print('Bagging results:')
print(DTC.__class__.__name__+" accuracy is %2.5f" % bag_DTC.score(X_test, y_test))

# show classification report of DTC algorithm
print(classification_report(y_test, y_pred_DTC))

# visualization: decision tree
print('Decision Tree:')
fig = plt.figure(figsize = (25,20))
_ = tree.plot_tree(DTC,   
                   feature_names = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'Cabin', 'male', 'C', 'Q', 'S'],
                   max_depth = 3,
                   filled = True)


# Random Forest Classifier
RFC = RandomForestClassifier(n_estimators = 10, max_depth = 5)
RFC.fit(X_train, y_train)

# predicting the values for RFC
y_pred_RFC = RFC.predict(X_test)

# show train/test split results of RFC algorithm
print('Train/Test split results:')
print(RFC.__class__.__name__+" accuracy is %2.5f" % accuracy_score(y_test, y_pred_RFC))

# using cross validation for RFC algorithm
scores_accuracy = cross_val_score(RFC, X, Y, cv = 10, scoring = 'accuracy')

# show cross validation results of RFC algorithm
print('Cross Validation results:')
print(RFC.__class__.__name__ + " accuracy is %2.5f" % scores_accuracy.mean())

# using ensemble method (bagging) for RFC algorithm
bag_RFC = BaggingClassifier(
    base_estimator = RFC, 
    n_estimators = 10, 
    n_jobs = -1, 
    random_state = 0
    )

bag_RFC.fit(X_train, y_train)

# show ensemble method (bagging) results of RFC algorithm
print('Bagging results:')
print(RFC.__class__.__name__+" is %2.5f" % bag_RFC.score(X_test, y_test))

# show classification report of RFC algorithm
print(classification_report(y_test, y_pred_RFC))

# visualization: confusion matrix of RFC
print('Confusion Matrix:')
visualize_RFC = plot_confusion_matrix(RFC, X_test, y_test, cmap = plt.cm.Reds, values_format = 'd')