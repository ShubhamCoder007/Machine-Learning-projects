# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:19:44 2018

@author: Shubham Banerjee
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Adult_data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,14].values

#replacing categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labenc_x=LabelEncoder()
x[:,1]=labenc_x.fit_transform(x[:,1])
x[:,3]=labenc_x.fit_transform(x[:,3])
x[:,5]=labenc_x.fit_transform(x[:,5])
x[:,6]=labenc_x.fit_transform(x[:,6])
x[:,7]=labenc_x.fit_transform(x[:,7])
x[:,8]=labenc_x.fit_transform(x[:,8])
x[:,9]=labenc_x.fit_transform(x[:,9])
x[:,13]=labenc_x.fit_transform(x[:,13])
labenc_y=LabelEncoder()
y=labenc_y.fit_transform(y)

#removing the missing values
from sklearn.preprocessing import Imputer
impute=Imputer(missing_values="NaN",
strategy="mean", axis=0)
impute.fit(x)
x=impute.transform(x)

#splitting datasets into training and testing sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#creating the SVM model
from sklearn.naive_bayes import GaussianNB 
classifier = GaussianNB()
classifier.fit(x_train,y_train)

#predicting the results
y_pred = classifier.predict(x_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#calculating percentage accuracy
from sklearn.metrics import accuracy_score
percentage_accuracy = accuracy_score(y_test,y_pred)*100

