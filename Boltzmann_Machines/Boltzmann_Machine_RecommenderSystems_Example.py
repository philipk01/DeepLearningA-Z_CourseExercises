# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 08:11:06 2020

@author: phili


Data: MovieLens from GroupLens (https://grouplens.org/datasets/movielens/)



Question:
Predict upward or downward trend for the stock price for 01/2017

Approach:
Using an LSTM model, get trend instead of looking for a specific stock price prediction.


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

