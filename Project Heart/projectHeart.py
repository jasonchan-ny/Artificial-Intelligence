"""
Created on Tue Nov 23 20:48:18 2021

@author: jason , victor, richard
"""

# 1. import the modules needed
import pandas as pd
import numpy as np # data manipulation
import matplotlib.pyplot as plt # matplotlib is for drawing graphs
import matplotlib.colors as colors
from sklearn.utils import resample # downsample the dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale # scale and center data
from sklearn.svm import SVC # this will make a support vector machine for classificaiton
from sklearn.model_selection import GridSearchCV # this will do cross validation

# 1. load the data using pandas
df = pd.read_csv('processedCleveland.csv', header = None)

# 2. print the first 5 rows of the data set
print(df.head())

# 3. examine the output of 2

# 4. determine the datatype of each column
df.dtypes

# 5. print the unique values for these columns (ca , thal)
print(df.ca.unique())
print(df.thal.unique())

# 6. missing values in column ca
df.ca.isnull().sum()

# 7. missing values in column thal
df.thal.isnull().sum()

# 8. determine how many rows contain missing values
print(len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')]))

# 9. examine the output of 8

# 10. count the number of rows in the full dataset
len(df)

# 11. remove the rows with missing values
df = df.dropna()

# 12. verify that you removed the rows by printing the size of the dataset
len(df)

# 13. verify the unique function that ca and thal do not have missing values
df.ca.unique()
df.thal.unique()

# 14. split the data into dependent and independent variables
#  a)
x = df.iloc[:,:-1]
#  b)
y = df.iloc[:,-1]

# 15. print the head of both the x and y dataframes so that you can verify this worked correctly
print(x.head())
print(y.head())

# 16. verify that the columns that are categorical contains the correct data values
#     then use one hot encoding to separate the columns “cp”, “restecg”, “slope”. and “thal” 
#     so that each resulting column only contains “0s” and “1s”

x_encoded = pd.get_dummies(x, columns = [
    'cp',
    'restecg',
    'slope',
    'thal'
    ])

x_encoded.head()

# 17. see the different levels of heart disease
y.unique()

#     see someone that only has heart disease or not
y_not_zero_idx = y > 0
y[y_not_zero_idx] = 1
y.unique()

# 18. determine how many levels the tree needs in order to accurately predict on “new” data
x_train, x_test, y_train, y_test = train_test_split(x_encoded.values, y, test_size = 0.2, random_state = 0)
max_depths = [50, 100, 150, 200, 250]
for max_depth in max_depths:
    clf1 = DecisionTreeClassifier(max_depth = max_depth).fit(x_train,y_train)
    y_pred = clf1.predict(x_test)
    print('Accuracy Score of ', max_depth, 'is: ', accuracy_score(y_test, y_pred))
