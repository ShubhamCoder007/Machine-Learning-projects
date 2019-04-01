# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:11:42 2018

@author: Shubham Banerjee
"""


#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('user_knowledge.csv').iloc[:,:6]
x = dataset.values

#if imputer is required
flag = dataset.isnull().sum()

#applying encoding to the categorical datas
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labenc_x = LabelEncoder()
x[:,5] = labenc_x.fit_transform(x[:,5])

oneHotEnc = OneHotEncoder(categorical_features=[5])
x = oneHotEnc.fit_transform(x).toarray()

#removing the dummy variable trap
x = x[:,1:] 

#plotting the elbow
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10,
                    max_iter = 300, random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.xlabel('No. of clusters -->')
plt.ylabel('wcss -->')
plt.title('Elbow curve')
plt.show()

#fitting our kmeans with 5 clusters and predicting the clusters
kmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init = 10,
                max_iter = 300, random_state = 42)
y_kmeans = kmeans.fit_predict(x)
