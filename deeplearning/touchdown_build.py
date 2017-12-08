import playcaller_models
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import os
np.random.seed(1671)

#data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep) 
#train_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep + 'train' + os.sep + team + os.sep + 'offense' + os.sep ) 
#test_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep + 'test' + os.sep + team + os.sep + 'offense' + os.sep ) 
history_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'deeplearning' + os.sep + 'history' + os.sep) 
models_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'deeplearning' + os.sep + 'h5' + os.sep ) 

TD_CLASS = 2
#load training data from csv file into numpy Array, last column is class label
#Train = np.loadtxt(train_path + os.sep + team+'.offense.play.type.train.csv', delimiter = ',', skiprows = 1) #skip header/label row
tdown_train = np.loadtxt('touchdown.train.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
tdown_train_predictors = tdown_train[:,0:tdown_train.shape[1]-2] # get all rows, all columns except the last column                                               
train_tdown_class = tdown_train[:,tdown_train.shape[1]-1]
#Train_class = Train[:,Train.shape[1]-1] # get the last column (the class label)
#Train_Predictors = Train[:1000,0:Train.shape[1]-2] # get all rows, all columns except the last column                                               
#Train_class = Train[:1000,Train.shape[1]-1] # get the last column (the class label)


#load training data from csv file into numpy Array, last column is class label
#Test = np.loadtxt(test_path + os.sep + team+'.offense.play.type.test.csv', delimiter = ',', skiprows = 1) #skip header/label row
tdown_test = np.loadtxt('touchdown.test.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
tdown_test_Predictors = tdown_test[:,0:tdown_test.shape[1]-2] # get all rows, all columns except the last column                                               
test_tdown_class = tdown_test[:,tdown_test.shape[1]-1]
#Test_class = Test[:,Test.shape[1]-1] # get the last column (the class label)

#Train_class= np_utils.to_categorical(Train_class, NB_CLASSES)
#Test_class= np_utils.to_categorical(Test_class, NB_CLASSES)

train_tdown_class= np_utils.to_categorical(train_tdown_class, TD_CLASS)
test_tdown_class= np_utils.to_categorical(test_tdown_class, TD_CLASS)

NUM_PREDICTORS = tdown_train_predictors.shape[1] #number of features

DNNTouchdown,DNNTouchdown_history=playcaller_models.TouchdownDNN(tdown_train_predictors,train_tdown_class,NUM_PREDICTORS, TD_CLASS)
DNNTouchdown.save(models_path + os.sep + 'touchdown21_DNNTouchdown.h5')
playcaller_models.SaveHistory(DNNTouchdown_history, history_path + os.sep + 'touchdown2_DNNTouchdown.hist.txt')
playcaller_models.plotTrainingAcc(DNNTouchdown_history, 'Touchdown - DNN 3 Hidden Layers, .3 Dropout, Lasso Regression - Accuracy')
playcaller_models.plotTrainingLoss(DNNTouchdown_history, 'Touchdown - DNN 3 Hidden Layers, .3 Dropout, Lasso Regression - Loss')

playcaller_models.deepPredict(DNNTouchdown, tdown_test_Predictors, test_tdown_class, NUM_PREDICTORS, TD_CLASS)
