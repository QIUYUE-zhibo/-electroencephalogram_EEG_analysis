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

y = data['Label']
data = data.drop(['Label','TimeStamp','TimeStamp_Readable','seconds_of_timestamp','CQ_AF3','CQ_T7','CQ_Pz','CQ_T8','CQ_AF4'], axis = 1)

data.head()

#dropping the un-named Column
data = data.drop(data.columns[0], axis=1)

x = data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=0)
#X_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)
#print("Shape of x_val :", x_val.shape)
#print("Shape of y_val :", y_val.shape)

#AdaBoosting Algorithm.
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy',max_depth = 7)#parameter tuning, 200 decision trees of depth 8
AdaBoost = AdaBoostClassifier(base_estimator=model ,n_estimators=400, learning_rate=1)
#fitting the model
AdaBoost.fit(x_train,y_train)
predict=AdaBoost.predict(x_test)
print("Training Accuracy :", AdaBoost.score(x_train, y_train))
print("Testing Accuracy :", AdaBoost.score(x_test, y_test))
cr = classification_report(y_test, predict)
print(cr)

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
plot_confusion_matrix(y_test, predict, classes=class_names,
title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plot_confusion_matrix(y_test, predict, classes=class_names,
normalize=True,
title='Normalized confusion matrix')
plt.show()


from sklearn.externals import joblib
joblib.dump(AdaBoost,"DTB_joblib_model")
dtb = joblib.load("DTB_joblib_model")

# importing ML Explanability Libraries
#for purmutation importance
import eli5
from eli5.sklearn import PermutationImportance
#for SHAP values
#import shap ADAboosting is not compatiable with shap
#from pdpbox import pdp, info_plots #for partial plots

perm = PermutationImportance(AdaBoost, random_state = 0).fit(x_test,y_test)
eli5.show_weights(perm, feature_names = x_test.columns.tolist())











