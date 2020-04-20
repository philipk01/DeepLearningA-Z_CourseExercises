"""
Problem statement: 
Stock prices can be described as a Brownian motion and that mean future prices (states) are independent from the past prices (states). Brownian motion is a random process and therefore preditions of future sates are inherently impossible.

Data:
5 years of Google stock prices, 01/2012 - 12/2016

Question:
Predict upward or downward trend for the stock price for 01/2017

Approach:
Using an LSTM model, get trend instead of looking for a specific stock price prediction.

This algorithm improves upon RNN_LSTM_stock_prices by adding 3 more fields from the .csv training file
"""
##############################
# Part 1 - Preprocessing
##############################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# remove ','
cols = list(dataset_train)[1:6]
cols
dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0,len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(",","")
        
dataset_train = dataset_train.astype(float)

training_set = dataset_train.iloc[:, 0:5].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60:i, 0:5])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

############################
# Part 2 - Building the RNN
############################
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# # Fitting the RNN to the Training set
# regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)

# Fitting RNN to training set using Keras Callbacks.
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='improved_ex_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
history = regressor.fit(X_train, y_train, shuffle=True, epochs=20,
                        callbacks=[es, rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=32)
 
################################################################
# Part 3 - Making the predictions and visualising the results
################################################################
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# remove ','
cols = list(dataset_test)[1:6]
dataset_test = dataset_test[cols].astype(str)
for i in cols:
    for j in range(0,len(dataset_test)):
        dataset_test[i][j] = dataset_test[i][j].replace(",","")
        
dataset_test = dataset_test.astype(float)

test_set = dataset_test.iloc[:, 0:5].values

# Getting the predicted stock price for each financial day of 01/2017
dataset_total = pd.concat((dataset_train.iloc[:, 0:5], dataset_test.iloc[:, 0:5]), axis = 0)
dataset_total = dataset_total.astype(float)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = sc.fit_transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0:5])
    
X_test = np.array(X_test)

predicted_stock_price = regressor.predict(X_test)

# create empty table withc 5 fields
temp = np.zeros(shape=(len(predicted_stock_price), X_test.shape[2]))
# put the predicted values in the right field
temp[:,0] = predicted_stock_price[:,0]
# inverse transform and then select the right field
predicted_stock_price = sc.inverse_transform(temp)[:,0]

#########################################
# Plot prediction and real stock prices
#########################################
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()