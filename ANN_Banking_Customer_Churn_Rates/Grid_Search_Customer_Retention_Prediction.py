# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:44:36 2020

@author: phili
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import timeit

import pandas as pd
df = pd.read_csv('Churn_Modelling.csv')
pd.set_option('display.max_columns', None)

x = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

#transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(drop='first'), [1, 2])], remainder='passthrough')
t = [('cat', OneHotEncoder(drop='first'), [1, 2]), ('num', StandardScaler(), [0, 3, 4, 5, 6, 7, 8, 9])]

transformer = ColumnTransformer(t, remainder='passthrough')

# transform training data
x = transformer.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# BUILD MODEL with Grid search to find best params
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [10, 20, 30],
              'epochs': [100, 200, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

start = timeit.timeit()
print("Time start...")

grid_search = grid_search.fit(x_train, y_train)

end = timeit.timeit()
print(end - start)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

best_parameters

best_accuracy