# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:34:30 2018

@author: Shubham Banerjee
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('user_knowledge.csv').iloc[:,:6]
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,5].values
 
#applying encoding to the categorical datas
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labenc_y = LabelEncoder()
y = labenc_y.fit_transform(y)

"""
oneHotEnc = OneHotEncoder(categorical_features=[5])
y = oneHotEnc.fit_transform(y).toarray()
y=y.T
#removing the dummy variable trap
y = y[:,1:] """

#splitting datasets into training and testing sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.80,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#applying the logostic regression model
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train,y_train)

#creating the SVM model
#This model however achieved even higher score using linear kernel
from sklearn.svm import SVC
classifier_lsvm = SVC(kernel = 'linear', random_state = 0)
classifier_lsvm.fit(x_train,y_train)

#creating the SVM model
#This model however achieved even higher score using linear kernel
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_svm.fit(x_train,y_train)

#using the k-nn classifier
from sklearn.neighbors import KNeighborsClassifier
classifier_kn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier_kn.fit(x_train,y_train)

#creating the DecisionTree model
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier_dt.fit(x_train,y_train)

#creating the Random Forest model
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier_rf.fit(x_train,y_train)

#predicting the results
y_pred_lr = classifier_lr.predict(x_test)
y_pred_lsvc = classifier_lsvm.predict(x_test)
y_pred_svc = classifier_svm.predict(x_test)
y_pred_kn = classifier_kn.predict(x_test)
y_pred_dt = classifier_dt.predict(x_test)
y_pred_rf = classifier_rf.predict(x_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test,y_pred_lr)
cm_lsvc = confusion_matrix(y_test,y_pred_lsvc)
cm_svc = confusion_matrix(y_test,y_pred_svc)
cm_kn = confusion_matrix(y_test,y_pred_kn)
cm_dt = confusion_matrix(y_test,y_pred_dt)
cm_rf = confusion_matrix(y_test,y_pred_rf)

#percentage accuracy score
from sklearn.metrics import accuracy_score
percentage_accuracy_lr = accuracy_score(y_test,y_pred_lr)*100
percentage_accuracy_lsvc = accuracy_score(y_test,y_pred_lsvc)*100
percentage_accuracy_svc = accuracy_score(y_test,y_pred_svc)*100
percentage_accuracy_kn = accuracy_score(y_test,y_pred_kn)*100
percentage_accuracy_dt = accuracy_score(y_test,y_pred_dt)*100
percentage_accuracy_rf = accuracy_score(y_test,y_pred_rf)*100


