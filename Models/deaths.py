# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:42:41 2020

@author: Leeza
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset_train = pd.read_csv('India_death_cases.csv')
training_set = dataset_train.iloc[61:134, 1:2].values


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_train = []
y_train = []
for i in range(5, 73):
    X_train.append(training_set_scaled[i-5:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 4, batch_size = 2)


dataset_test = pd.read_csv('India_death_cases.csv')
real_death_case = dataset_train.iloc[:134, 1:2].values


dataset_total = pd.concat((dataset_train['India'], dataset_test['India']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) + 5:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(5, 209):
    X_test.append(inputs[i-5:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_death_case = regressor.predict(X_test)
predicted_death_case = sc.inverse_transform(predicted_death_case)
predicted_death_case = predicted_death_case.astype(int)


# Visualising the results
plt.plot(real_death_case, color = 'red', label = 'Real Covid-19 Death Cases')
plt.plot(predicted_death_case, color = 'blue', label = 'Predicted Covid-19 Death Cases')
plt.title('Covid-19 Death Cases')
plt.xlabel('Time')
plt.ylabel('Total no. of Deaths')
plt.legend(loc='best')
plt.show()
