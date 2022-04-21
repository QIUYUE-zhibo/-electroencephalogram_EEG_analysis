

def warn(*args, **kwargs):
 pass
 import warnings
 warnings.warn = warn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from keras.models import Model, Sequential
from keras.models import Sequential
#from keras.layers import Input, Dense
from keras.layers import Dense
import keras.layers as layers
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
import itertools

data = pd.read_csv('Overall_Training_Data_v0.1.csv')

data = data.sample(frac=1).reset_index(drop=True)

data.head(5)

Train_size = int(len(data)* .8)
print("Train size: %d" % Train_size)
Test_size = len(data) - Train_size
print("Test size: %d" % Test_size)

inputdata = data[['AF3_THETA', 'AF3_ALPHA', 'AF3_LOW_BETA','AF3_HIGH_BETA','AF3_GAMMA','T7_THETA',
'T7_ALPHA','T7_LOW_BETA','T7_HIGH_BETA','T7_GAMMA',
'Pz_THETA','Pz_ALPHA','Pz_LOW_BETA','Pz_HIGH_BETA',
'Pz_GAMMA','T8_THETA','T8_ALPHA','T8_LOW_BETA',
'T8_HIGH_BETA','T8_GAMMA','AF4_THETA','AF4_ALPHA',
'AF4_LOW_BETA','AF4_HIGH_BETA','AF4_GAMMA']]

target = data['Label']

target.shape

inputdata = np.array(inputdata, dtype=float)
target = np.array(target, dtype=float)

#np.set_printoptions(precision=3)

#np.array2string(inputdata1,threshold=np.inf, max_line_width=np.inf,separator=',').replace('\n', '')

#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=25)

print (inputdata)

#np.set_printoptions(threshold=np.inf)
#print(target)

#np.reshape(inputdata, (len(data),25,1))

inputdata.shape

target.shape
x_train, x_test, y_train, y_test = train_test_split(inputdata, target, test_size=0.4, random_state = 4)

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.4, random_state = 4)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
x_validate = np.reshape(x_validate, (x_validate.shape[0], 1, x_validate.shape[1]))

model = Sequential()

#np.reshape(x_train, (len(x_train),25,1))
#np.reshape(x_test, (Test_size,25,1))

model.add( LSTM((1), batch_input_shape=(None,1,25), return_sequences=True))
model.add( LSTM((1), return_sequences=False))

#First Trial of Network
#model.add( LSTM((1), batch_input_shape=(None,1,25), return_sequences=True))
#model.add( LSTM((1), batch_input_shape=(None,1,25), return_sequences=True))
#model.add( LSTM((1), return_sequences=False))

model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])

model.summary()

#history = model.fit(x_train,y_train, epochs=500, validation_data=(x_test, y_test))
#history = model.fit(x_train,y_train, epochs=1000, validation_data=(x_test, y_test))
history = model.fit(x_train,y_train, epochs=200)

skill = model.evaluate(x_validate, y_validate, steps=100)

#history = model.fit(x_train,y_train, epochs=50)
history = model.fit(x_train,y_train, epochs=200, validation_data=(
x_test, y_test))

#skill = model.evaluate(x_test, y_test, steps=100)

# list all data in history
print(history.history.keys())

#result = model.predict_proba(x_test)
result = model.predict(x_test)

len(result)
result

len(y_test)
#y_test

plt.scatter(range(len(result)),result,c='r')
plt.scatter(range(len(y_test)),y_test,c='g')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plt.plot(history.history['loss'])
#plt.show()

#y_test = np.array(y_test)
#y_test
#Result = np.argmax(result, axis=1)
#print(Result)

result = model.predict_classes(x_test)

#cn = confusion_matrix(y_test, result)
#print(confusion_matrix(np.argmax(y_test), Result))

def plot_confusion_matrix(y_test, result, classes, normalize=False, title=None, cmap=plt.cm.Blues):
 """
431 This function prints and plots the confusion matrix.
432 Normalization can be applied by setting ‘normalize=True‘.
433 """
 if not title:
  if normalize:
    title = 'Normalized confusion matrix'
  else:
    title = 'Confusion matrix, without normalization'

 # Compute confusion matrix
 cm = confusion_matrix(y_test, result)
# Only use the labels that appear in the data
# classes = classes[unique_labels(y_test, result)]
 if normalize:
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print("Normalized confusion matrix")
 else:
  print('Confusion matrix, without normalization')
 print(cm)

 fig, ax = plt.subplots()
 im = ax.imshow(cm, interpolation='nearest', cmap = cmap)
 ax.figure.colorbar(im, ax=ax)
 # We want to show all ticks...

 ax.set(xticks=np.arange(cm.shape[1]),
  yticks = np.arange(cm.shape[0]),
  # ... and label them with the respective list entries
  xticklabels = classes, yticklabels = classes,
  title = title,
  ylabel ='True label',
  xlabel ='Predicted label')

 # Rotate the tick labels and set their alignment.
 plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
 rotation_mode = "anchor")

 # Loop over data dimensions and create text annotations.
 fmt = '.2f' if normalize else 'd'
 thresh = cm.max() / 2.
 for i in range(cm.shape[0]):
     for j in range(cm.shape[1]):
         ax.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
 fig.tight_layout()
 return ax

np.set_printoptions(precision=2)

class_names = ['Low Risk', 'Low Medium', 'Medium Risk', 'High Risk']
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, result, classes=class_names,title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, result, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()

AF3_THETA = data['AF3_THETA'][:Train_size]
AF3_ALPHA = data['AF3_ALPHA'][:Train_size]
AF3_LOW_BETA = data['AF3_LOW_BETA'][:Train_size]
AF3_HIGH_BETA = data['AF3_HIGH_BETA'][:Train_size]
AF3_GAMMA =  data['AF3_GAMMA'][:Train_size]

T7_THETA = data['T7_THETA'][:Train_size] 
T7_ALPHA = data['T7_ALPHA'][:Train_size] 
T7_LOW_BETA = data['T7_LOW_BETA'][:Train_size] 
T7_HIGH_BETA = data['T7_HIGH_BETA'][:Train_size] 
T7_GAMMA = data['T7_GAMMA'][:Train_size]

PZ_THETA = data['Pz_THETA'][:Train_size] 
PZ_ALPHA = data['Pz_ALPHA'][:Train_size] 
PZ_LOW_BETA = data['Pz_LOW_BETA'][:Train_size] 
PZ_HIGH_BETA = data['Pz_HIGH_BETA'][:Train_size] 
PZ_GAMMA = data['Pz_GAMMA'][:Train_size]

T8_THETA = data['T8_THETA'][:Train_size]
T8_ALPHA = data['T8_ALPHA'][:Train_size]
T8_LOW_BETA = data['T8_LOW_BETA'][:Train_size]
T8_HIGH_BETA = data['T8_HIGH_BETA'][:Train_size]
T8_GAMMA = data['T8_GAMMA'][:Train_size]

AF4_THETA = data['AF4_THETA'][:Train_size]
AF4_ALPHA = data['AF4_ALPHA'][:Train_size]
AF4_LOW_BETA = data['AF4_LOW_BETA'][:Train_size]
AF4_HIGH_BETA = data['AF4_HIGH_BETA'][:Train_size]
AF4_GAMMA = data['AF4_GAMMA'][:Train_size]


Label = data['Label'][:Train_size]

AF3_THETA = data['AF3_THETA'][Train_size:]
AF3_ALPHA = data['AF3_ALPHA'][Train_size:]
AF3_LOW_BETA = data['AF3_LOW_BETA'][Train_size:]
AF3_HIGH_BETA = data['AF3_HIGH_BETA'][Train_size:]
AF3_GAMMA = data['AF3_GAMMA'][Train_size:]

T7_THETA = data['T7_THETA'][Train_size:]
T7_ALPHA = data['T7_ALPHA'][Train_size:]
T7_LOW_BETA = data['T7_LOW_BETA'][Train_size:]
T7_HIGH_BETA = data['T7_HIGH_BETA'][Train_size:]
T7_GAMMA = data['T7_GAMMA'][Train_size:]

PZ_THETA = data['Pz_THETA'][Train_size:]
PZ_ALPHA = data['Pz_ALPHA'][Train_size:]
PZ_LOW_BETA = data['Pz_LOW_BETA'][Train_size:]
PZ_HIGH_BETA = data['Pz_HIGH_BETA'][Train_size:]
PZ_GAMMA = data['Pz_GAMMA'][Train_size:]

T8_THETA = data['T8_THETA'][Train_size:]
T8_ALPHA = data['T8_ALPHA'][Train_size:]
T8_LOW_BETA = data['T8_LOW_BETA'][Train_size:]
T8_HIGH_BETA = data['T8_HIGH_BETA'][Train_size:]
T8_GAMMA = data['T8_GAMMA'][Train_size:]

AF4_THETA = data['AF4_THETA'][Train_size:]
AF4_ALPHA = data['AF4_ALPHA'][Train_size:]
AF4_LOW_BETA = data['AF4_LOW_BETA'][Train_size:]
AF4_HIGH_BETA = data['AF4_HIGH_BETA'][Train_size:]
AF4_GAMMA =data['AF4_GAMMA'][Train_size:]

Label = data['Label'][Train_size:]

AF3_THETA_input = layers.Input(shape=(1,))
AF3_ALPHA_input = layers.Input(shape=(1,))
AF3_LOW_BETA_input = layers.Input(shape=(1,))
AF3_HIGH_BETA_input = layers.Input(shape=(1,))
AF3_GAMMA_input = layers.Input(shape=(1,))
T7_THETA_input = layers.Input(shape=(1,))
T7_ALPHA_input = layers.Input(shape=(1,))
T7_LOW_BETA_input = layers.Input(shape=(1,))
T7_HIGH_BETA_input = layers.Input(shape=(1,))
T7_GAMMA_input = layers.Input(shape=(1,))
PZ_THETA_input = layers.Input(shape=(1,))
PZ_ALPHA_input = layers.Input(shape=(1,))
PZ_LOW_BETA_input = layers.Input(shape=(1,))
PZ_HIGH_BETA_input = layers.Input(shape=(1,))
PZ_GAMMA_input = layers.Input(shape=(1,))
T8_THETA_input = layers.Input(shape=(1,))
T8_ALPHA_input = layers.Input(shape=(1,))
T8_LOW_BETA_input = layers.Input(shape=(1,))
T8_HIGH_BETA_input = layers.Input(shape=(1,))
T8_GAMMA_input = layers.Input(shape=(1,))
AF4_THETA_input = layers.Input(shape=(1,))
AF4_ALPHA_input = layers.Input(shape=(1,))
AF4_LOW_BETA_input = layers.Input(shape=(1,))
AF4_HIGH_BETA_input = layers.Input(shape=(1,))
AF4_GAMMA_input =layers.Input(shape=(1,))

merge_layer = layers.concatenate([AF3_THETA_input, AF3_ALPHA_input,
AF3_LOW_BETA_input, AF3_HIGH_BETA_input, AF3_GAMMA_input,
T7_THETA_input, T7_ALPHA_input, T7_LOW_BETA_input,
T7_HIGH_BETA_input,
T7_GAMMA_input, PZ_THETA_input, PZ_ALPHA_input, PZ_LOW_BETA_input,
PZ_HIGH_BETA_input, PZ_GAMMA_input, T8_THETA_input, T8_ALPHA_input,
T8_LOW_BETA_input, T8_HIGH_BETA_input, T8_GAMMA_input,
AF4_THETA_input,
AF4_ALPHA_input, AF4_LOW_BETA_input, AF4_HIGH_BETA_input,
AF4_GAMMA_input ])

import keras.models as Model
merge_layer = layers.Dense(256, activation='relu')(merge_layer)
predictions = layers.Dense(1)(merge_layer)
wide_model = Model(inputs=[AF3_THETA_input, AF3_ALPHA_input,
AF3_LOW_BETA_input, AF3_HIGH_BETA_input, AF3_GAMMA_input,
T7_THETA_input, T7_ALPHA_input, T7_LOW_BETA_input,
T7_HIGH_BETA_input,
T7_GAMMA_input, PZ_THETA_input, PZ_ALPHA_input, PZ_LOW_BETA_input,
PZ_HIGH_BETA_input, PZ_GAMMA_input, T8_THETA_input, T8_ALPHA_input,
T8_LOW_BETA_input, T8_HIGH_BETA_input, T8_GAMMA_input,
AF4_THETA_input,
AF4_ALPHA_input, AF4_LOW_BETA_input, AF4_HIGH_BETA_input,AF4_GAMMA_input ], outputs = predictions)

wide_model.compile(loss='mean_absolute_error',optimizer='adam',
metrics=['accuracy'])
wide_model.summary()












