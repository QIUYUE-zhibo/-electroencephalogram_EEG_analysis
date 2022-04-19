import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
x = pd.DataFrame(x, columns = data.columns)

x.shape
y.shape

from sklearn.model_selection import train_test_split

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

x_train = x_train.reshape(1240, 5, 5, 1)
x_test = x_test.reshape(310, 5, 5, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), input_shape=(5,5,1), strides=(1, 1), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (2, 2), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.summary()

batch_size = 10
epochs = 1500
history = model.fit(x_train, y_train_binary, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test_binary))
result = model.predict_classes(x_test)

def plot_confusion_matrix(y_test, result, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    """
     This function prints and plots the confusion matrix.
    Normalization can be applied by setting ‘normalize=True‘.
     """
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
           yticks=np.arange(cm.shape[0]),
    # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

class_names = ['Low Risk', 'Low Medium', 'Medium Risk', 'High Risk']
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, result, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, result, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()

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


