#network.py
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(36, input_dim=18, activation='hard_sigmoid'))
model.add(Dense(36, activation='hard_sigmoid'))
#model.add(Dense(36, activation='hard_sigmoid'))
#model.add(Dense(36, activation='hard_sigmoid'))
model.add(Dense(9, activation='hard_sigmoid'))
model.compile(optimizer='rmsprop',loss='mse')

import numpy as np
	
vector = np.random.random((9967,18))

#set vectors
f = open("/home/minki/Documents/vectorInput.txt",'r')
for i in range(0,9967):
	tempString = f.readline()
	for j in range(0,18):
		vector[i][j] = tempString[j]
f.close()

answer = np.random.random((9967,9))

#set answers
f = open("/home/minki/Documents/answer.txt",'r')
for i in range(0,9967):
	tempString = f.readline()
	for j in range(0,9):
		answer[i][j] = tempString[j]
f.close()

model.fit(vector,answer, epochs=100, batch_size=1)

model.save("/home/minki/Documents/model.h5")