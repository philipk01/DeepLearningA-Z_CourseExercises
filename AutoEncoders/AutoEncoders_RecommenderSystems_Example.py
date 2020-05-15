# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 08:11:06 2020

@author: phili

Data: MovieLens from GroupLens (https://grouplens.org/datasets/movielens/)
we need 2 data direcotries:
    1. ml-1m
    2. ml-100k

Question:
predict user ratings 1 - 5

Approach:
Implement a Stacked AutoEncoders model with PyTorch

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

import time

# f_path = '/home/nezo/AI/DeepLearningA-Z_HandsOnCourse_CP/Boltzmann_Machines/'
f_path = 'C:\\Users\\phili\\main\\git\\DeepLearningA-Z_HandsOnCourse\\Boltzmann_Machines\\'

# movies = pd.read_csv(f_path + 'ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# users = pd.read_csv(f_path + 'ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# ratings = pd.read_csv(f_path + 'ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# # Preparing the training set and the test set - k-test fold left for autoencoders, hence only 1 set
# training_set = pd.read_csv(f_path + 'ml-100k/u1.base', delimiter = '\t')
# training_set = np.array(training_set, dtype = 'int')
# test_set = pd.read_csv(f_path + 'ml-100k/u1.test', delimiter = '\t')
# test_set = np.array(test_set, dtype = 'int')

# large dataset: 1 million inputs
training_set = pd.read_csv(f_path + 'ml-1m\\training_set.csv')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv(f_path + 'ml-1m\\test_set.csv')
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

# Sparse Autoencoder
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        # Architecture full connection of NN
        self.fc1 = nn.Linear(nb_movies, 20) # object of the linear class; tunable value 20
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        # activation function
        self.activation = nn.Sigmoid() # Tunable 

    def forward(self, x):
        # encoding
        x = self.activation(self.fc1(x)) # first encoded vector
        x = self.activation(self.fc2(x))
        # decoding, 10 elements to 20
        x = self.activation(self.fc3(x))
        # decode to get reconstructed input vector
        x = self.fc4(x) # vector of predicted ratings
        return x

class SAE_2(nn.Module):
    def __init__(self, ):
        super(SAE_2, self).__init__()
        # Architecture full connection of NN
        self.fc1 = nn.Linear(nb_movies, 25) # object of the linear class; tunable value 20
        self.fc2 = nn.Linear(25, 15)
        self.fc3 = nn.Linear(15, 10)
        self.fc4 = nn.Linear(10, 15)
        self.fc5 = nn.Linear(15, 25)
        self.fc6 = nn.Linear(25, nb_movies)
        # activation function
        self.activation = nn.Sigmoid() # Tunable 

    def forward(self, x):
        # encoding
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        # decode to get reconstructed input vector
        x = self.fc6(x) # vector of predicted ratings
        return x


nb_epoch = 5
########################################### 
# Model 1, 4 layers
########################################### 
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # try Adam

start_time = time.time()

# Train
# nb_epoch = 3
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # additional dim corresponding to a btach, here we have a batch containing a single input vector
        target = input.clone()
        temp = torch.sum(target.data > 0)
        if temp > 0: # to optimize memory, use only users who rated >= 1 movie
            output = sae(input) # vector of predicted ratings
            target.require_grad = False # gradient NOT computed, saves memory
            output[target == 0] = 0 # to save memory, include only originally rated movies
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(temp + 1e-10) # nb_movies / nb_movies with positive ratings; +1e-10 to guaraentee that this sum != 0
            loss.backward() # direction for update weights
            train_loss += np.sqrt(loss.item() * mean_corrector) # item(): part of loss object that contains error
            s += 1. # number of users raitng at least 1 movie
            optimizer.step() # update weights; intensity of updates to weights
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))
    
print("--- %s seconds ---" % (time.time() - start_time))

# Test
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    temp = torch.sum(target.data > 0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(temp + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))


###########################################   
# Model 2, 6 layers
########################################### 
# Train    
sae = SAE_2()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # try Adam

start_time = time.time()

# Training the SAE
# nb_epoch = 15
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # additional dim corresponding to a btach, here we have a batch containing a single input vector
        target = input.clone()
        temp = torch.sum(target.data > 0)
        if temp > 0: # to optimize memory, use only users who rated >= 1 movie
            output = sae(input) # vector of predicted ratings
            target.require_grad = False # gradient NOT computed, saves memory
            output[target == 0] = 0 # to save memory, include only originally rated movies
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(temp + 1e-10) # nb_movies / nb_movies with positive ratings; +1e-10 to guaraentee that this sum != 0
            loss.backward() # direction for update weights
            train_loss += np.sqrt(loss.item() * mean_corrector) # item(): part of loss object that contains error
            s += 1. # number of users raitng at least 1 movie
            optimizer.step() # update weights; intensity of updates to weights
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))
    
print("--- %s seconds ---" % (time.time() - start_time))
    
    
# Test
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    temp = torch.sum(target.data > 0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(temp + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

    
###########################################   
## GPU
###########################################   
# nb_epoch = 15

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

########################################### 
# Model 1, 4 layers
########################################### 
m1 = torch.cuda.memory_allocated()

sae = SAE().to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # try Adam

start_time = time.time()

# Train
# nb_epoch = 3
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0).to(device) # additional dim corresponding to a btach, here we have a batch containing a single input vector
        target = input.clone().to(device)
        temp = torch.sum(target.data > 0).to(device)

#         input = input.to(device)
#         input = target.to(device)
#         temp = temp.to(device)
        
        if temp > 0: # to optimize memory, use only users who rated >= 1 movie
            output = sae(input) # vector of predicted ratings
            target.require_grad = False # gradient NOT computed, saves memory
            output[target == 0] = 0 # to save memory, include only originally rated movies
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(temp + 1e-10) # nb_movies / nb_movies with positive ratings; +1e-10 to guaraentee that this sum != 0
            loss.backward() # direction for update weights
            train_loss += np.sqrt(loss.item() * mean_corrector) # item(): part of loss object that contains error
            s += 1. # number of users raitng at least 1 movie
            optimizer.step() # update weights; intensity of updates to weights
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))
    
print("--- %s seconds ---" % (time.time() - start_time))


m2 = torch.cuda.memory_allocated()

print("total memory for Model 1 on GPU: ", m2 - m1)

# Test
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    temp = torch.sum(target.data > 0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(temp + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

###########################################   
# Model 2, 6 layers
########################################### 
torch.cuda.empty_cache()
# Train    
sae = SAE_2().to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # try Adam

m3 = torch.cuda.memory_allocated()

start_time = time.time()

# Training the SAE
# nb_epoch = 15
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0).to(device) # additional dim corresponding to a btach, here we have a batch containing a single input vector
        target = input.clone().to(device)
        temp = torch.sum(target.data > 0).to(device)
        if temp > 0: # to optimize memory, use only users who rated >= 1 movie
            output = sae(input).to(device) # vector of predicted ratings
            target.require_grad = False # gradient NOT computed, saves memory
            output[target == 0] = 0 # to save memory, include only originally rated movies
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(temp + 1e-10) # nb_movies / nb_movies with positive ratings; +1e-10 to guaraentee that this sum != 0
            loss.backward() # direction for update weights
            train_loss += np.sqrt(loss.item() * mean_corrector) # item(): part of loss object that contains error
            s += 1. # number of users raitng at least 1 movie
            optimizer.step() # update weights; intensity of updates to weights
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))
    
print("--- %s seconds ---" % (time.time() - start_time))
    
m4 = torch.cuda.memory_allocated()

print("m3", m3, "\nm4", m4, "\ntotal memory for Model 2 on GPU: ", m4 - m3)
    
# Test
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0).to(device)
    target = Variable(test_set[id_user]).unsqueeze(0).to(device)
    temp = torch.sum(target.data > 0).to(device)
    if torch.sum(target.data > 0) > 0:
        output = sae(input).to(device)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(temp + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss / s))
