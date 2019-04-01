# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:02:00 2018

@author: nEW u
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Car.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,6].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
X[:,1]=labelencoder_X.fit_transform(X[:,1])
X[:,4]=labelencoder_X.fit_transform(X[:,4])
X[:,5]=labelencoder_X.fit_transform(X[:,5])

onehotencoder = OneHotEncoder(categorical_features =[0,1,4,5])
X= onehotencoder.fit_transform(X).toarray()
labenc_Y= LabelEncoder()
Y = labenc_Y.fit_transform(Y)

#Splitting data sets into training and testing sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=0)#preferably random state value is used as 0

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#applying the logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#predict result
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
