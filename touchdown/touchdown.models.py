import deeplearning 
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import os

NB_CLASSES = 2 #number of classes

data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep) 
history_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'history' + os.sep + 'touchdown') 
models_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'h5' + os.sep + 'touchdown') 

#load training data from csv file into numpy Array, last column is class label
Train = np.loadtxt(data_path + os.sep + 'touchdown.train.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
#Train_Predictors = Train[:,0:Train.shape[1]-2] # get all rows, all columns except the last column                                               
#Train_class = Train[:,Train.shape[1]-1] # get the last column (the class label)
Train_Predictors = Train[:1000,0:Train.shape[1]-2] # get all rows, all columns except the last column                                               
Train_class = Train[:1000,Train.shape[1]-1] # get the last column (the class label)


#load training data from csv file into numpy Array, last column is class label
Test = np.loadtxt(data_path + os.sep + 'touchdown.test.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
Test_Predictors = Test[:,0:Train.shape[1]-2] # get all rows, all columns except the last column                                               
Test_class = Test[:,Train.shape[1]-1] # get the last column (the class label)

Train_class= np_utils.to_categorical(Train_class, NB_CLASSES)
Test_class= np_utils.to_categorical(Test_class, NB_CLASSES)

NUM_PREDICTORS = 136 #number of features

#logistic model
logistic_model, logistic_history = deeplearning.LogisticRegression(Train_Predictors, Train_class, NUM_PREDICTORS, NB_CLASSES)
logistic_model.save(models_path + os.sep + 'touchdown_logistic.h5')
deeplearning.SaveHistory(logistic_history, history_path + os.sep +'touchdown_logistic.hist.txt')

#DNN model
DNN_model, DNN_history = deeplearning.DeepNN(Train_Predictors, Train_class, NUM_PREDICTORS, NB_CLASSES)
DNN_model.save(models_path + os.sep +'touchdown_DNN.h5')
deeplearning.SaveHistory(DNN_history, history_path + os.sep + 'touchdown_DNN.hist.txt')

DNNDropOut_model,DNNDropOut_history=deeplearning.DeepNNDropOut(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
DNNDropOut_model.save(models_path + os.sep +'touchdown_DNNDropOut.h5')
deeplearning.SaveHistory(DNNDropOut_history, history_path + os.sep + 'touchdown_DNNDropOut.hist.txt')

DNNDOMaxNorm_model,DNNDOMaxNorm_history=deeplearning.DeepNNDropOutMaxNorm(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
DNNDOMaxNorm_model.save(models_path + os.sep + 'touchdown_DNNDOMaxNorm.h5')
deeplearning.SaveHistory(DNNDOMaxNorm_history, history_path + os.sep + 'touchdown_DNNDOMaxNorm.hist.txt')

DNNLasso,DNNLasso_history=deeplearning.DeepNNLasso(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
DNNLasso.save(models_path + os.sep + 'touchdown_DNNLasso.h5')
deeplearning.SaveHistory(DNNLasso_history, history_path + os.sep + 'touchdown_DNNLasso.hist.txt')

DNNRidge,DNNRidge_history=deeplearning.DeepNNRidge(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
DNNRidge.save(models_path + os.sep + 'touchdown_DNNRidge.h5')
deeplearning.SaveHistory(DNNRidge_history, history_path + os.sep + 'touchdown_DNNRidge.hist.txt')

DNNElastic,DNNElastic_history=deeplearning.DeepNNElastic(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
DNNElastic.save(models_path + os.sep + 'touchdown_DeepNNElastic.h5')
deeplearning.SaveHistory(DNNElastic_history, history_path + os.sep + 'touchdown_DNNElastic.hist.txt')

import matplotlib.pyplot as plt

#f1=plt.figure(200)
#deeplearning.plotTrainingAcc(logistic_history,'logistic')
#deeplearning.plotTrainingLoss(logistic_history,'logistic')
#deeplearning.plotTrainingAcc(DNN_history,'DeepNN')
#deeplearning.plotTrainingLoss(DNN_history,'DeepNN')
#deeplearning.plotTrainingAcc(DNNDropOut_history,'DeepNNDropOut')
#deeplearning.plotTrainingLoss(DNNDropOut_history,'DeepNNDropOut')
#deeplearning.plotTrainingAcc(DNNLasso_history,'DeepNNLasso')
#deeplearning.plotTrainingLoss(DNNLasso_history,'DeepNNLasso')
#deeplearning.plotTrainingAcc(DNNRidge_history,'DeepNNRidge')
#deeplearning.plotTrainingLoss(DNNRidge_history,'DeepNNRidge')
#deeplearning.plotTrainingAcc(DNNElastic_history,'DeepNNElastic')
#deeplearning.plotTrainingLoss(DNNElastic_history,'DeepNNElastic')
#f1.savefig("touchdown_acc.pdf", bbox_inches='tight')