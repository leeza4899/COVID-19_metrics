# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:43:15 2020

@author: Leeza
"""

#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset_train = pd.read_csv('india_rate.csv')
training_set = dataset_train.iloc[0:133, 1:2].values

#scaling the training set
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#appending the train data to X_train
X_train = []
y_train = []
for i in range(5, 133):
    X_train.append(training_set_scaled[i-5:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping the X_tran
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#import modules from keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialize the regressor
regressor = Sequential()

#input layer of model
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#hidden layers
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#output layer of the model
regressor.add(Dense(units = 1))

#compile the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fit the train dataset to the model
regressor.fit(X_train, y_train, epochs = 5, batch_size = 2)

#loading the test dataset
dataset_test = pd.read_csv('india_rate.csv')
real_death_case = dataset_train.iloc[:133, 1:2].values

#predicting the future data
dataset_total = pd.concat((dataset_train['New Recovery'], dataset_test['New Recovery']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) + 5:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(5, 208):
    X_test.append(inputs[i-5:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_recovery_rate = regressor.predict(X_test)
predicted_recovery_rate  = sc.inverse_transform(predicted_recovery_rate )


# Visualising the results
plt.plot(predicted_recovery_rate , color = 'blue', label = 'Predicted Covid-19 Recovery Rates')
plt.title('Covid-19 Recovery Rate')
plt.xlabel('Time')
plt.ylabel('Rate')
plt.legend(loc='best')
plt.show()
