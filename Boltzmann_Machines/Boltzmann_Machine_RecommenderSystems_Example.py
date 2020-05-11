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

Note: the data preprocessing step below will also be used for Autoencoders part exercise
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable # for Stochastic Gradient Descent

f_path = '/home/nezo/AI/DeepLearningA-Z_HandsOnCourse_CP/Boltzmann_Machines/'

# not used in this example, here only to show the data content
movies = pd.read_csv(f_path + 'ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv(f_path + 'ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv(f_path + 'ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set - k-test fold left for autoencoders, hence only 1 set used here - u1
# Columns: users Id - movie Id - rating values (1-5) - timestamp (not used)
training_set = pd.read_csv(f_path + 'ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv(f_path + 'ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies to create an nb_users x nb_movies matrix of ratings
# this way allows us to use other train-test data that might contain different values
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = [] # to become a list of lists - a list for each user
    for id_users in range(1, nb_users + 1): # user_id starts at 1
        id_movies = data[:,1][data[:,0] == id_users] # indices of rated movies
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

# view user_id = 1, at index 0
training_set[0]

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# view user_id = 1, at index 0
training_set[0]

# Create the Probabilistic Graphical Model
# create the architecture of the Neural Network - the RBM
class RBM():
    def __init__(self, nv, nh): # number of nodes
        
        # initialize weight and bias, the parameters that will be optimized
        self.W = torch.randn(nh, nv)
        
        # initialize bias for hidden nodes (h) and visible nodes (v)
        # bias of hidden nodes: p(h | v)
        self.a = torch.randn(1, nh) # 1 dim for batch
        
        # bias of the visible nodes: p(v | h)
        self.b = torch.randn(1, nv) # 1 dim for batch
        
    # Sample hidden nodes - for Gibbs sampling to approximate the Log Likelihood Gradient
    # This is the sigmoid activatoin function applied to: wx + b
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) # wx
        
        # wx + b
        activation = wx + self.a.expand_as(wx) # expand dim of bias to match wx
        
        p_h_given_v = torch.sigmoid(activation)
        
        # Setting RBM as Bernoulli RBM; sampling p(h = 1 | v)
        return p_h_given_v, torch.bernoulli(p_h_given_v) # return bernoulli samples of h according to p_h_given_v; vector of 0, 1
     
    # Sample visible nodes - for Gibbs sampling to approximate the Log Likelihood Gradient    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        
        activation = wy + self.b.expand_as(wy) # apply bias to each batch
        
        p_v_given_h = torch.sigmoid(activation)
        
        # Setting RBM as Bernoulli RBM; sampling p(v = 1 | h)
        return p_v_given_h, torch.bernoulli(p_v_given_h) # return bernoulli samples of v according to p_v_given_h; vector of 0, 1
        
    # Contrastive Divergence: approximate Log Likelihood Gradients by optimizing weights
    # v0: input vector of ratings by a user
    # vk: visible nodes after k iterations of CD
    # ph0: vector of probabilities at first iteration where p(h = 1 | v0)
    # phk: vector of probabilities at iteration k where p(h = 1 | vk)
    def train(self, v0, vk, ph0, phk):  
        
        # Update weight parameters
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        
        # Update bias of p_v_given_h
        self.b += torch.sum((v0 - vk), 0) # 0, to keep 2 dim
        
        # Update bias of p_h_given_v
        self.a += torch.sum((ph0 - phk), 0) # 0, to keep 2 dim            
            
nv = len(training_set[0])

# Tunable parameters
nh = 150 # number of features to detect
batch_size = 10
nb_epoch = 5
cd_steps = 10 # steps in Contrastive Divergence

# Creat RBM Oject!!!!
rbm = RBM(nv, nh)

# Training the RBM
trials = np.zeros([nb_epoch, 2])
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # counter to normalize train_loss
    
    # Training steps
    for id_user in range(0, nb_users - batch_size, batch_size):

        # output of Gibbs sampling, initial vector in Gibbs chain that will be updated over k iterations
        # vk: last sample of random walk
        vk = training_set[id_user : id_user + batch_size]

        # initial ratings
        v0 = training_set[id_user : id_user + batch_size]

        # initial probabilities
        ph0, _ = rbm.sample_h(v0)
        
        # k steps of CD - Gibbs chain, i.e., k steps of random walk
        for k in range(cd_steps):
            _, hk = rbm.sample_h(vk) # sampling of hidden nodes
            _, vk = rbm.sample_v(hk) # update vk
            vk[v0 < 0] = v0[v0 < 0] # don't update -1 ratings, ratings are either -1, 0, 1
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk) # update weights
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch: '+ str(epoch) + ' loss: ' + str(train_loss / s))
    a[epoch - 1][0] = epoch
    a[epoch - 1][1] = train_loss / s

opt_index = np.where(trials == min(trials[:, 1]))[0][0]
         
opt_epoch = int(trials[opt_index][0])
min_accuracy = a[min_index][1]

print('Optimal epoch number', min_epoch, '\nBest accuracy', round(min_accuracy, 4) * 100, '%')



# Test the RBM model
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user : id_user + 1] # to activate neurons of RBM
    vt = test_set[id_user : id_user + 1]

    # k steps of CD - Gibbs chain, i.e., k steps of random walk
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h) # predicted ratings
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
print('Test loss: ' + str(test_loss / s))


       
class train_test():
    def __init__(self, training_set, test_set, nb_epoch, batch_size, nb_users, cd_steps):
        self.training_set = training_set
        self.test_set = test_set
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.nb_users = nb_users
        self.cd_steps = cd_steps
    
    def train(self):
        a = []
        # Training the RBM
        for epoch in range(1, self.nb_epoch + 1):
            train_loss = 0
            s = 0. # counter to normalize train_loss

            # Training steps
            for id_user in range(0, self.nb_users - self.batch_size, self.batch_size):

                # output of Gibbs sampling, initial vector in Gibbs chain that will be updated over k iterations
                # vk: last sample of random walk
                vk = self.training_set[id_user : id_user + self.batch_size]

                # initial ratings
                v0 = self.training_set[id_user : id_user + self.batch_size]

                # initial probabilities
                ph0, _ = rbm.sample_h(v0)

                # k steps of CD - Gibbs chain, i.e., k steps of random walk
                for k in range(cd_steps):
                    _, hk = rbm.sample_h(vk) # sampling of hidden nodes
                    _, vk = rbm.sample_v(hk) # update vk
                    vk[v0 < 0] = v0[v0 < 0] # don't update -1 ratings, ratings are either -1, 0, 1
                phk, _ = rbm.sample_h(vk)
                rbm.train(v0, vk, ph0, phk) # update weights
                train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
                s += 1.
            print('epoch: '+ str(epoch) + ' loss: ' + str(train_loss / s))
            
            return epoch, test_loss / s

            
    def test(self):
        # Test the RBM model
        test_loss = 0
        s = 0.
        for id_user in range(self.nb_users):
            v = self.training_set[id_user : id_user + 1] # to activate neurons of RBM
            vt = self.test_set[id_user : id_user + 1]

            # k steps of CD - Gibbs chain, i.e., k steps of random walk
            if len(vt[vt >= 0]) > 0:
                _, h = rbm.sample_h(v)
                _, v = rbm.sample_v(h) # predicted ratings
                test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
                s += 1.
        print('Test loss: ' + str(test_loss / s))
        return test_loss / s

            
t = train_test(training_set = training_set, test_set = test_set, nb_epoch = 10, batch_size = 100, nb_users = nb_users, cd_steps = 10)
t.train()
t.test()


parameters = {'batch_size': [10, 20, 30],
              'epochs': [100, 200, 500]}

parameters['batch_size']

m = []
for i in parameters['batch_size']:
    for j in parameters['epochs']:
        t = train_test(training_set = training_set, test_set = test_set, nb_epoch = 10, batch_size = 100, nb_users = nb_users, cd_steps = 10)
        t.train()
        
