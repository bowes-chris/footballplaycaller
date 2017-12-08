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

#FD_CLASS = 2
#load training data from csv file into numpy Array, last column is class label
#Train = np.loadtxt(train_path + os.sep + team+'.offense.play.type.train.csv', delimiter = ',', skiprows = 1) #skip header/label row
yards_gained_train = np.loadtxt('yards.gained.train.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
yards_gained_train_predictors = yards_gained_train[:,0:yards_gained_train.shape[1]-2] # get all rows, all columns except the last column                                               
train_yards_gained_class = yards_gained_train[:,yards_gained_train.shape[1]-1]
#Train_class = Train[:,Train.shape[1]-1] # get the last column (the class label)
#Train_Predictors = Train[:1000,0:Train.shape[1]-2] # get all rows, all columns except the last column                                               
#Train_class = Train[:1000,Train.shape[1]-1] # get the last column (the class label)


#load training data from csv file into numpy Array, last column is class label
#Test = np.loadtxt(test_path + os.sep + team+'.offense.play.type.test.csv', delimiter = ',', skiprows = 1) #skip header/label row
yards_gained_test = np.loadtxt('yards.gained.test.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
yards_gained_test_Predictors = yards_gained_test[:,0:yards_gained_test.shape[1]-2] # get all rows, all columns except the last column                                               
test_yards_gained_class = yards_gained_test[:,yards_gained_test.shape[1]-1]
#Test_class = Test[:,Test.shape[1]-1] # get the last column (the class label)

#Train_class= np_utils.to_categorical(Train_class, NB_CLASSES)
#Test_class= np_utils.to_categorical(Test_class, NB_CLASSES)

#train_yards_gained_class= np_utils.to_categorical(train_yards_gained_class, FD_CLASS)
#test_yards_gained_class= np_utils.to_categorical(test_yards_gained_class, FD_CLASS)

NUM_PREDICTORS = yards_gained_train_predictors.shape[1] #number of features

DNNYardsGained,DNNYardsGained_history=playcaller_models.YardsGainedDNN(yards_gained_train_predictors,train_yards_gained_class,NUM_PREDICTORS)
DNNYardsGained.save(models_path + os.sep + 'yards.gained1_DNNPlaytype.h5')
playcaller_models.SaveHistory(DNNYardsGained_history,  history_path + os.sep  + 'yards.gained1_DNNPlaytype.hist.txt')
playcaller_models.plotTrainingAcc(DNNYardsGained_history, 'Yards Gained - DNN 3 Hidden Layers, .3 Dropout, Ridge Regression - Accuracy')
playcaller_models.plotTrainingLoss(DNNYardsGained_history, 'Yards Gained - DNN 3 Hidden Layers, .3 Dropout, Ridge Regression - Loss')
#playcaller_models.deepPredict(DNNYardsGained, yards_gained_test_Predictors, test_yards_gained_class, NUM_PREDICTORS, FD_CLASS)
