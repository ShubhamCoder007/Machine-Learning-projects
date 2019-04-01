# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:51:01 2018

@author: Shubham Banerjee
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('diabetes_data.csv')
x = dataset.iloc[:,[3,4]].values
