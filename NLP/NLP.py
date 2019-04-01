# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:15:11 2018

@author: Shubham Banerjee
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv',sep='\t',quoting=3)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating our bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#splitting datasets into training and testing sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.90,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#creating the Naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(x_train,y_train)

#creating the DecisionTree model
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier_dt.fit(x_train,y_train)

#creating the Random Forest model
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier_rf.fit(x_train,y_train)

#predicting the results
y_pred_dt = classifier_dt.predict(x_test)
y_pred_rf = classifier_rf.predict(x_test)
y_pred_nb = classifier_nb.predict(x_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test,y_pred)
cm_dt = confusion_matrix(y_test,y_pred)
cm_rf = confusion_matrix(y_test,y_pred)

#percentage accuracy score
from sklearn.metrics import accuracy_score
percentage_accuracy_nb = accuracy_score(y_test,y_pred_nb)*100
percentage_accuracy_dt = accuracy_score(y_test,y_pred_dt)*100
percentage_accuracy_rf = accuracy_score(y_test,y_pred_rf)*100