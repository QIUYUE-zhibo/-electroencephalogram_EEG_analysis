import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

data = pd.read_csv('Overall_Training_Data_v0.1.csv')

#Checkng the shape of the data
data.shape

#Example of the data
data.head()

#information about each attribute
data.info()

data.describe()

data.isnull().sum()

#Shuffling the data for better results
data=data.reindex(np.random.permutation(data.index))

data.head()

y = data['Label']
data = data.drop(['Label','TimeStamp','TimeStamp_Readable','seconds_of_timestamp'], axis = 1)

data.head()

#dropping the un-named Column
data = data.drop(data.columns[0], axis=1)

x = data

x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=0)

print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)

#K Nearest Neighbour
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
predict=model.predict(x_test)
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
print(confusion_matrix(y_test,predict))
cr = classification_report(y_test, predict)
print(cr)




