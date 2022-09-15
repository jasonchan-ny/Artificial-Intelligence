#!/usr/bin/env python
# coding: utf-8

# In[198]:


# import the modules needed
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier


# In[199]:


# load train set
train_df = pd.read_csv('train.csv')


# In[200]:


# show the dimensions of train set
print(train_df.shape)


# In[201]:


# show the first 5 rows in train set
print(train_df.head(5))


# In[202]:


# exploratory data analysis
print(train_df.info())
print(train_df.describe())


# In[203]:


# load test set
test_df = pd.read_csv('test.csv')


# In[204]:


# show the dimensions of test set
print(test_df.shape)


# In[205]:


# boolean: embarkment
for k in train_df.Embarked.unique():
    if type(k) == str:
        train_df['emb_' + k] = (train_df.Embarked == k) * 1


# In[206]:


# boolean: is_male
train_df['is_male'] = (train_df.Sex == 'male') * 1


# In[207]:


# boolean: has_cabin
train_df.loc[:, 'has_cabin'] = 0
train_df.loc[train_df.Cabin.isna(), 'has_cabin'] = 1


# In[208]:


# fill in the missing values for age
train_df.loc[train_df.Age.isna(), 'Age'] = 100


# In[209]:


# model preprocessing
features = train_df[["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]]
Y = train_df["Survived"]
#X = train_df[["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]]
#X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)


# In[210]:


# one hot encoding
one_hot_encoded_training_predictors = pd.get_dummies(features)
one_hot_encoded_training_predictors.head()
X = one_hot_encoded_training_predictors


# In[211]:


# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1)


# In[212]:


# Logistic Regression Algorithm
LR = LogisticRegression(max_iter = 1000) 
#solver = 'lbfgs' , solver = liblinear
LR.fit(X_train, y_train)


# In[213]:


# predicting the values for LR
y_pred_LR = LR.predict(X_test)


# In[214]:


# show train/test split results for LR algorithm
print('Train/Test split results:')
print(LR.__class__.__name__+" accuracy is %2.5f" % accuracy_score(y_test, y_pred_LR))

#print(LR.__class__.__name__+" accuracy is: {:}" .format(LR.score(X_test, y_test)))


# In[215]:


# using cross validation for LR algorithm
scores_accuracy = cross_val_score(LR, X, Y, cv=10, scoring = 'accuracy')


# In[216]:


# show cross validation results of LR algorithm
print('Cross Validation results:')
print(LR.__class__.__name__+" accuracy is %2.5f" % scores_accuracy.mean())


# In[217]:


# using ensemble method (bagging) for LR algorithm
bag_LR = BaggingClassifier(
    base_estimator = LR, 
    n_estimators = 10, 
    n_jobs = -1, 
    random_state = 0
    )

bag_LR.fit(X_train, y_train)


# In[218]:


# show ensemble method (bagging) results of LR algorithm
print('Bagging results:')
#print(bag_LR.score(X_test, y_test))
print(LR.__class__.__name__+" accuracy is %2.5f" % bag_LR.score(X_test, y_test))


# In[219]:


# Decision Tree Classifier
DTC = DecisionTreeClassifier(max_depth = 5)
DTC.fit(X_train, y_train)


# In[220]:


# predicting the values for DTC
y_pred_DTC = DTC.predict(X_test)


# In[221]:


# show train/test split results of DTC algorithm
print('Train/Test split results:')
print(DTC.__class__.__name__+" accuracy is %2.5f" % accuracy_score(y_test, y_pred_DTC))


# In[222]:


# using cross validation for DTC algorithm
scores_accuracy = cross_val_score(DTC, X, Y, cv = 10, scoring = 'accuracy')


# In[223]:


# show cross validation results of DTC algorithm
print('Cross Validation results:')
print(DTC.__class__.__name__ + " accuracy is %2.5f" % scores_accuracy.mean())


# In[224]:


# using ensemble method (bagging) for DTC algorithm
bag_DTC = BaggingClassifier(
    base_estimator = DTC, 
    n_estimators = 10, 
    n_jobs = -1, 
    random_state = 0
    )

bag_DTC.fit(X_train, y_train)


# In[225]:


# show ensemble method (bagging) results of DTC algorithm
print('Bagging results:')
print(DTC.__class__.__name__+" accuracy is %2.5f" % bag_DTC.score(X_test, y_test))


# In[226]:


# Random Forest Classifier
RFC = RandomForestClassifier(n_estimators = 10, max_depth = 5)
RFC.fit(X_train, y_train)


# In[227]:


# predicting the values for RFC
y_pred_RFC = RFC.predict(X_test)


# In[228]:


# show train/test split results of RFC algorithm
print('Train/Test split results:')
print(RFC.__class__.__name__+" accuracy is %2.5f" % accuracy_score(y_test, y_pred_RFC))


# In[229]:


# using cross validation for RFC algorithm
scores_accuracy = cross_val_score(RFC, X, Y, cv = 10, scoring = 'accuracy')


# In[230]:


# show cross validation results of RFC algorithm
print('Cross Validation results:')
print(RFC.__class__.__name__ + " accuracy is %2.5f" % scores_accuracy.mean())


# In[231]:


# using ensemble method (bagging) for RFC algorithm
bag_RFC = BaggingClassifier(
    base_estimator = RFC, 
    n_estimators = 10, 
    n_jobs = -1, 
    random_state = 0
    )

bag_RFC.fit(X_train, y_train)


# In[232]:


# show ensemble method (bagging) results of RFC algorithm
print('Bagging results:')
print(RFC.__class__.__name__+" is %2.5f" % bag_RFC.score(X_test, y_test))


# In[ ]:




