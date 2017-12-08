import playcaller_models
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import os
np.random.seed(1671)

def plot_roc(y_test, y_score, title):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' - (AUC = %0.2f)' % roc_auc)
    plt.show()
    print('AUC: %f' % roc_auc)

history_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'deeplearning' + os.sep + 'history' + os.sep) 
models_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'deeplearning' + os.sep + 'h5' + os.sep ) 

FD_CLASS = 2

fdown_test = np.loadtxt('first.down.test.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
fdown_test_Predictors = fdown_test[:,0:fdown_test.shape[1]-2] # get all rows, all columns except the last column                                               
fdown_test_class = fdown_test[:,fdown_test.shape[1]-1]

firstdown_model = load_model(models_path + os.sep + 'first.down_DNNPlaytype.h5')
firstdown_score = firstdown_model.predict(fdown_test_Predictors)
plot_roc(fdown_test_class, np.argmax(firstdown_score, axis=1), 'First Down ROC Analysis')

fdown_test_class= np_utils.to_categorical(fdown_test_class, FD_CLASS)
playcaller_models.deepPredict(firstdown_model, fdown_test_Predictors, fdown_test_class, fdown_test_Predictors.shape[1], FD_CLASS)
#######
tdown_test = np.loadtxt('touchdown.test.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
tdown_test_Predictors = tdown_test[:,0:tdown_test.shape[1]-2] # get all rows, all columns except the last column                                               
test_tdown_class = tdown_test[:,tdown_test.shape[1]-1]

touchdown_model = load_model(models_path + os.sep + 'touchdown1_DNNTouchdown.h5')
touchdown_score = touchdown_model.predict(tdown_test_Predictors)
plot_roc(test_tdown_class, np.argmax(touchdown_score, axis=1), 'Touchdown ROC Analysis')

#########
pldg_test = np.loadtxt('play.type.PLDG.test.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
pldg_test_Predictors = pldg_test[:,0:pldg_test.shape[1]-5] # get all rows, all columns except the last column
dir_test_Predictors = pldg_test[:,0:pldg_test.shape[1]-4] # get all rows, all columns except the last column
test_pldg_class = pldg_test[:,pldg_test.shape[1]-4]
test_len_class = pldg_test[:,pldg_test.shape[1]-3]
test_dir_class = pldg_test[:,pldg_test.shape[1]-2]
test_gap_class = pldg_test[:,pldg_test.shape[1]-1]

pldg_model = load_model(models_path + os.sep + 'pldg1_DNNPlaytype.h5')
pldg_score = pldg_model.predict(pldg_test_Predictors)
plot_roc(test_pldg_class, np.argmax(pldg_score, axis=1), 'Play Type ROC Analysis')

####### need to do multiple-classification ROC for these, may not have time
dir_model = load_model(models_path + os.sep + 'play.direction2_DNNPlaytype.h5')
dir_score = dir_model.predict(pldg_test[:,0:pldg_test.shape[1]-4])
plot_roc(test_dir_class, np.argmax(dir_score, axis=1), 'Dir ROC Analysis')

length_model = load_model(models_path + os.sep + 'play.length2_DNNPlaytype.h5')
length_score = length_model.predict(pldg_test[:,0:pldg_test.shape[1]-3])
length_roc(test_len_class, np.argmax(length_score, axis=1), 'Length Type ROC Analysis')

gap_model = load_model(models_path + os.sep + 'play.gap2_DNNPlaytype.h5')
gap_score = gap_model.predict(pldg_test[:,0:pldg_test.shape[1]-3])
plot_roc(test_gap_class, np.argmax(gap_score, axis=1), 'Gap Type ROC Analysis')