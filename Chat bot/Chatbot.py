# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:34:19 2018

@author: Shubham Banerjee
"""

import pandas as pd
import re

dataset = pd.read_csv('chatdata.csv',encoding='latin-1')

convo = dataset.iloc[:,1]

clist = []

def qa_pairs(x):
    cpairs = re.findall(": (.*?)(?:$|\\n)",x)
    clist.extend(list(zip(cpairs,cpairs[1:])))

convo.map(qa_pairs)
convo_frame = pd.Series(dict(clist)).to_frame().reset_index()
convo_frame.columns = ['q','a']

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer(ngram_range=(1,3))
vec = vectorizer.fit_transform(convo_frame['q'])

'''
my_q = vectorizer.transform(["Hi. my name is Shubham"])
cs = cosine_similarity(my_q,vec)
rs = pd.Series(cs[0]).sort_values(ascending=False)
top5 = rs.iloc[0:5]

rsi = convo_frame['q'][top5.index]
'''

def get_response(q):
    my_q = vectorizer.transform([q])
    cs = cosine_similarity(my_q, vec)
    rs = pd.Series(cs[0]).sort_values(ascending = False)
    return convo_frame.iloc[rs.index[0]]['a']

def bot_chat():
	while True:
		msg = str(input('Enter your question : '))
		get_response(msg)
        if 'good bye' in msg:
			print("\nBye :)")
			break
			
bot_chat()




