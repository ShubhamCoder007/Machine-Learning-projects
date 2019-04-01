# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:36:35 2018

@author: Shubham Banerjee
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import json
from difflib import get_close_matches
data=json.load(open("data.json")).keys()

def correct_word(word):
    if word in data:
        return word
    elif len(get_close_matches(word,data))>0:
        return get_close_matches(word,data)[0].lower()
    else:
        return word

#importing the data set
dataset = pd.read_csv('twitter_globalwarm.csv')
dataset['existence'] = dataset['existence'].replace('Y','Yes')
dataset['existence'] = dataset['existence'].replace('N','No') 

#null datas in the dataset
nulls = dataset.isnull().sum()

#cleaning the dataset for useful information
tweet = []
ex = []
for i in range(0,len(dataset)):
    if str(dataset['existence'][i]) != 'nan':
        tweet.append([dataset['tweet'][i]])
        ex.append([dataset['existence'][i]])
    
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0,len(tweet)):
    review = re.sub('[^a-zA-Z]',' ',str(tweet[i]))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    review = review.rstrip('link')
    corpus.append(review)
    
corpus_new = []
for i in range(0,10):
    w = corpus[i].split()
    w = [correct_word(word) for word in w]
    corpus_new.append(w)


#Creating our bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 8150)
x = cv.fit_transform(corpus).toarray()
y = ex


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
cm_nb = confusion_matrix(y_test,y_pred_nb)
cm_dt = confusion_matrix(y_test,y_pred_dt)
cm_rf = confusion_matrix(y_test,y_pred_rf)

#percentage accuracy score
from sklearn.metrics import accuracy_score
percentage_accuracy_nb = accuracy_score(y_test,y_pred_nb)*100
percentage_accuracy_dt = accuracy_score(y_test,y_pred_dt)*100
percentage_accuracy_rf = accuracy_score(y_test,y_pred_rf)*100
