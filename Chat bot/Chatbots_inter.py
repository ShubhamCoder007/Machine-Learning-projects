# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:46:50 2018

@author: Shubham Banerjee
"""

import pandas as pd
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataset = pd.read_csv('chatdata.csv',encoding='latin-1')

convo = dataset.iloc[:,1]

clist = []

def qa_pairs(x):
    cpairs = re.findall(": (.*?)(?:$|\\n)",x)
    clist.extend(list(zip(cpairs,cpairs[1:])))

convo.map(qa_pairs)
convo_frame = pd.Series(dict(clist)).to_frame().reset_index()
convo_frame.columns = ['q','a']

vectorizer = TfidfVectorizer(ngram_range=(1,3))
vec = vectorizer.fit_transform(convo_frame['q'])

class Bots:
    bot_count=0
    creator = 'Master Shubham Banerjee'
    def __init__(self,name):
        self.name = name
        Bots.bot_count+=1
        #vectorizer = TfidfVectorizer(ngram_range=(1,3))
        #vec = vectorizer.fit_transform(convo_frame['q'])
        self.thisbot = {'what is your name?':self.name,
                        'who is your creator?':Bots.creator,
						Bots.creator:'Oh! he created me too... I love him'}
    
    def introduce(self):
        return 'Hi, my name is '+str(self.name)+'\n'+Bots.creator+' created me.'
        
    def get_response(self,q):
        if q in self.thisbot:
            return thisbot[q]
        my_q = vectorizer.transform([q])
        cs = cosine_similarity(my_q, vec)
        rs = pd.Series(cs[0]).sort_values(ascending = False)
        return convo_frame.iloc[rs.index[0]]['a']

    
sean = Bots('Sean')
april = Bots('April')

bot=[sean,april]

def bot_chat(bot):
    while True:
        i=0
        msg = bot[i].introduce()
        while True:
            print(bot[i%Bots.bot_count].name+' : '+msg)
            msg = bot[i%Bots.bot_count].get_response(msg)
            i=i+1
            time.sleep(3)
            if i==chat_count:
                break
        break

chat_count = int(input('Enter the chat count = '))
bot_chat(bot)
