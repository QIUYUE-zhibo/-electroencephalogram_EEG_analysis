import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

data = pd.read_csv('Overall_Training_Data_v0.1.csv')

data.shape#Checkng the shape of the data
data.head()#Example of the data
#information about each attribute
data.info()
data.describe()
data.isnull().sum()
#Shuffling the data for better results
data=data.reindex(np.random.permutation(data.index))
data.head()
y = data['Label']

data = data.drop(['Label','TimeStamp','TimeStamp_Readable','seconds_of_timestamp','CQ_AF3','CQ_T7','CQ_Pz','CQ_T8','CQ_AF4'], axis = 1)
data.head()

#dropping the un-named Column
data = data.drop(data.columns[0], axis=1)

x = data
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

x.shape
y.shape

x_train, x_test, y_train, y_test = train_test_split( np.asarray(x),
np.asarray(y), test_size=0.2, random_state=0,shuffle= True)

print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)

from tensorflow import keras

# The known number of output classes.
num_classes = 5

# Input image dimensions
input_shape = (25,)

# Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(1240,25,1)
x_test = x_test.reshape(310,25,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D

model = Sequential()
model.add(Conv1D(32, (5), input_shape=(25,1), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='softmax'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])

model.summary()

batch_size = 10
epochs = 1500
model = model.fit(x_train, y_train_binary,
batch_size=batch_size,
epochs=epochs,
verbose=1,
validation_data=(x_test, y_test_binary))

