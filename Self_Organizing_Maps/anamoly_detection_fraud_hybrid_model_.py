# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:51:31 2020

@author: phili

Data: credit card application data

Problem: identify patterns in nonlinear high dimensional data and one of these nonlinear relationships would be fraud.

Question: find the probability of fraud per customers

Approach: Create a hybrid deep learning model composed of supervised and unsupervised models to predict fraud.
"""

##################################################
# Part 1 - Identify Fraudulent account using SOMs
##################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values

# not labeled data but rather for distinction between customers who were approved and those not approved
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = X.shape[1], sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['2', '1'] # [1] for NOT approved
colors = ['r', 'y']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X) # dictionary

# to get fraudulent accounts, get coordinates of white square from SOM map above
# frauds = mappings[(7, 2)]

# concatenate when more than one outlier detected in SOM
frauds = np.concatenate((mappings[2 ,1], mappings[3, 1], mappings[(4, 2)], mappings[(8, 8)]), axis = 0)

frauds = sc.inverse_transform(frauds)

###########################################################
# Part 2 - Unupervised --> Supervised deep learning
# create dependant variable for the supervised model - ANN
###########################################################
fid = frauds[:, 0:1].reshape(-1) # CustomerID of Frauds
# fid = fid.reshape(-1)

df = dataset.sort_values(by = ['CustomerID'], ascending=True)

df_frauds = df[df['CustomerID'].isin(fid)]

