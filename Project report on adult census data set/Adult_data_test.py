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
dataset2 = pd.read_csv('adult_data_test.csv')
x_train = dataset.iloc[:,:-1].values
y_train = dataset.iloc[:,14].values
x_test = dataset2.iloc[:,:-1].values
y_test = dataset2.iloc[:,14].values

#replacing categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labenc_x=LabelEncoder()
x_train[:,1] = labenc_x.fit_transform(x_train[:,1])
x_train[:,3] = labenc_x.fit_transform(x_train[:,3])
x_train[:,5] = labenc_x.fit_transform(x_train[:,5])
x_train[:,6] = labenc_x.fit_transform(x_train[:,6])
x_train[:,7] = labenc_x.fit_transform(x_train[:,7])
x_train[:,8] = labenc_x.fit_transform(x_train[:,8])
x_train[:,9] = labenc_x.fit_transform(x_train[:,9])
x_train[:,13] = labenc_x.fit_transform(x_train[:,13])

x_test[:,1] = labenc_x.fit_transform(x_test[:,1])
x_test[:,3] = labenc_x.fit_transform(x_test[:,3])
x_test[:,5] = labenc_x.fit_transform(x_test[:,5])
x_test[:,6] = labenc_x.fit_transform(x_test[:,6])
x_test[:,7] = labenc_x.fit_transform(x_test[:,7])
x_test[:,8] = labenc_x.fit_transform(x_test[:,8])
x_test[:,9] = labenc_x.fit_transform(x_test[:,9])
x_test[:,13] = labenc_x.fit_transform(x_test[:,13])

labenc_y=LabelEncoder()
y_test = labenc_y.fit_transform(y_test)
y_train = labenc_y.fit_transform(y_train)

#removing the missing values
from sklearn.preprocessing import Imputer
impute=Imputer(missing_values="NaN",
strategy="mean", axis=0)
impute.fit(x_train)
x_train = impute.transform(x_train)
impute.fit(x_test)
x_test = impute.transform(x_test)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#creating the GaussianNB model
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

