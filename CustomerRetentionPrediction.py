# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:21:50 2019

@author: phili
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encode Country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# encode Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# dummy variables; categorical variables are not ordinale, i.e., no relational order between the categories
# create dummy variables
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X).toarray()
# remove first dummy variable to avoid dummy variable trap
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split # obsolete
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2
import keras
# 2 modules needed
# Sequential model; to initialize ANN
from keras.models import Sequential
# Dense module to build ANN layers
from keras.layers import Dense # initializes weights

# initialize ANN; classificaiton problem
classifier = Sequential() # ANN model

# Add hidden layers
# first hidden layer
#classifier.add(Dense(units = 6, init = 'glorot_uniform', activation = 'relu', ))
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim = 11))

# second hidden layer
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit ANN to training set; connects ANN with data
# classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# validate model on test set using Con. Mat.
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Model accuracy: ", round((1491+225)/(2000), 3)*100, '%')


"""
predict if the customer with the following informations will leave the bank: 

    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40 years old
    Tenure: 3 years
    Balance: $60000
    Number of Products: 2
    Does this customer have a credit card ? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $50000
    So should we say goodbye to that customer ?

"""


new_prediction = classifier.predict(sc.fit_transform(np.array([[0, 0, 600, 1,  40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
