# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 08:11:06 2020

@author: phili


Data: MovieLens from GroupLens (https://grouplens.org/datasets/movielens/)
we need 2 data direcotries:
    1. ml-1m
    2. ml-100k

Question:
predict whether a user did not like a movie (input 0) or liked a movie (input 1)

Approach:
Use Boltzmann Machine to make the binary predictions

Note: the data preprocessing step below will also be used for Autoencoders part of the course

"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set - k-test fold left for autoencoders, hence only 1 set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = [] # to become a list of lists - a list for each user
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convert all ratings to binary input 0 (not liked), 1 (liked), from 1-5, for consistency between input and output
# value -1 for not rated movies
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Create the Probabilistic Graphical Model
# create the architecture of the Neural Network - the RBM
class RBM():
    def __init__(self, nv, nh):
        # initialize weight and bias, the parameters that will be optimized
        self.W = torch.randn(nh, nv)
        # initialize bias for hidden and visible nodes; p(h | v)
        self.a = torch.randn(1, nh) # bias of hidden nodes: p(hidden nodes | visible nodes)
        self.b = torch.randn(1, nv) # bias of the visible nodes: p(v | h)
        
        # sampling p(h = 1 | v) - Gibbs sampling to approximate the log likelihood gradient
    def sample_h(self, x):
        # p(h | v)
        wx = torch.mm(x, self.W.t())
        
        # activation function
        activation = wx + self.a.expand_as(wx) # apply bias to each batch
        
        # probability h is activated given v
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # return bernoulli samples of h since samples are binary
        
    def sample_v(self, y): # y corresponds to h
        # p(h | v)
        wy = torch.mm(y, self.W)
        
        # activation function
        activation = wy + self.b.expand_as(wy) # apply bias to each batch
        
        # probability h is activated given v
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h) # return bernoulli samples of h since samples are binary
        
      # Maximize Log Likelihood of the training set -> approximate gradients
    def train(self, v0, vk, ph0, phk): # CD-K
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0 - vk), 0) # 0, to keep 2 dim
        self.a += torch.sum((ph0 - phk), 0) # 0, to keep 2 dim            
            
nv = len(training_set[0])
nh = 100 # number of features to detect - tunable
batch_size = 100 # tunable
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):

        # output of Gibbs sampling
        vk = training_set[id_user:id_user+batch_size]

        # initial ratings
        v0 = training_set[id_user:id_user+batch_size]

        # initial probabilities
        ph0,_ = rbm.sample_h(v0)
        
        # k steps of CD - Gibbs chain, i.e., k steps of random walk
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0] # don't update -1 ratings
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Test the RBM model
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1] # to activate neurons of RBM
    vt = test_set[id_user:id_user + 1]

    # k steps of CD - Gibbs chain, i.e., k steps of random walk
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h) # predicted ratings
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
print('Test loss: '+str(test_loss/s))