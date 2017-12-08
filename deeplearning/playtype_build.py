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

PTYPE_CLASS = 13
PLDG_CLASS = 2
LEN_CLASS = 3
DIR_CLASS = 3
GAP_CLASS = 4

#load training data from csv file into numpy Array, last column is class label
#Train = np.loadtxt(train_path + os.sep + team+'.offense.play.type.train.csv', delimiter = ',', skiprows = 1) #skip header/label row
ptype_train = np.loadtxt('play.type.train.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
ptype_train_predictors = ptype_train[:,0:ptype_train.shape[1]-2] # get all rows, all columns except the last column
train_ptype_class = ptype_train[:,ptype_train.shape[1]-1]

pldg_train = np.loadtxt('play.type.PLDG.train.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
pldg_train_predictors = ptype_train[:,0:ptype_train.shape[1]-5] # get all rows, all columns except the last column                                               
train_pldg_class = pldg_train[:,pldg_train.shape[1]-4]
train_len_class = pldg_train[:,pldg_train.shape[1]-3]
train_dir_class = pldg_train[:,pldg_train.shape[1]-2]
train_gap_class = pldg_train[:,pldg_train.shape[1]-1]
#Train_class = Train[:,Train.shape[1]-1] # get the last column (the class label)
#Train_Predictors = Train[:1000,0:Train.shape[1]-2] # get all rows, all columns except the last column                                               
#Train_class = Train[:1000,Train.shape[1]-1] # get the last column (the class label)


#load training data from csv file into numpy Array, last column is class label
#Test = np.loadtxt(test_path + os.sep + team+'.offense.play.type.test.csv', delimiter = ',', skiprows = 1) #skip header/label row
ptype_test = np.loadtxt('play.type.test.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
ptype_test_Predictors = ptype_test[:,0:ptype_test.shape[1]-2] # get all rows, all columns except the last column
test_ptype_class = ptype_test[:,ptype_test.shape[1]-1]

pldg_test = np.loadtxt('play.type.PLDG.test.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
pldg_test_Predictors = pldg_test[:,0:pldg_test.shape[1]-5] # get all rows, all columns except the last column
test_pldg_class = pldg_test[:,pldg_test.shape[1]-4]
test_len_class = pldg_test[:,pldg_test.shape[1]-3]
test_dir_class = pldg_test[:,pldg_test.shape[1]-2]
test_gap_class = pldg_test[:,pldg_test.shape[1]-1] # get the last column (the class label)
#Test_class = Test[:,Test.shape[1]-1] # get the last column (the class label)

#Train_class= np_utils.to_categorical(Train_class, NB_CLASSES)
#Test_class= np_utils.to_categorical(Test_class, NB_CLASSES)

train_ptype_class= np_utils.to_categorical(train_ptype_class, PTYPE_CLASS)
test_ptype_class= np_utils.to_categorical(test_ptype_class, PTYPE_CLASS)

train_pldg_class= np_utils.to_categorical(train_pldg_class, PLDG_CLASS)
train_len_class= np_utils.to_categorical(train_len_class, LEN_CLASS)
train_dir_class= np_utils.to_categorical(train_dir_class, DIR_CLASS)
train_gap_class= np_utils.to_categorical(train_gap_class, GAP_CLASS)

test_pldg_class= np_utils.to_categorical(test_pldg_class, PLDG_CLASS)
test_len_class= np_utils.to_categorical(test_len_class, LEN_CLASS)
test_dir_class= np_utils.to_categorical(test_dir_class, DIR_CLASS)
test_gap_class= np_utils.to_categorical(test_gap_class, GAP_CLASS)

NUM_PREDICTORS = ptype_train_predictors.shape[1] #number of features
PLDG_PREDICTORS = pldg_train_predictors.shape[1]

DNNPlaytype,DNNPlaytype_history=playcaller_models.PlayTypeDNN(ptype_train_predictors,train_ptype_class,NUM_PREDICTORS, PTYPE_CLASS)
DNNPlaytype.save(models_path + os.sep + 'play.type_DNNPlaytype.h5')
playcaller_models.SaveHistory(DNNPlaytype_history,  history_path + os.sep  + 'play.type_DNNPlaytype.hist.txt')
playcaller_models.plotTrainingAcc(DNNPlaytype_history, 'Play Type - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Accuracy')
playcaller_models.plotTrainingLoss(DNNPlaytype_history, 'Play Type - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Loss')
deepPredict(DNNPlaytype, ptype_test_Predictors, test_ptype_class, NUM_PREDICTORS, PTYPE_CLASS)

DNNPldg,DNNPldg_history=playcaller_models.PlayTypeDNN(pldg_train[:,0:pldg_train.shape[1]-5],train_pldg_class,pldg_train[:,0:pldg_train.shape[1]-5].shape[1], PLDG_CLASS)
DNNPldg.save(models_path + os.sep + 'pldg2_DNNPlaytype.h5')
playcaller_models.SaveHistory(DNNPldg_history,  history_path + os.sep  + 'pldg2_DNNPlaytype.hist.txt')
playcaller_models.plotTrainingAcc(DNNPldg_history, 'Run/Pass - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Accuracy')
playcaller_models.plotTrainingLoss(DNNPldg_history, 'Run/Pass - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Loss')
playcaller_models.deepPredict(DNNPldg, pldg_test[:,0:pldg_test.shape[1]-5], test_pldg_class, pldg_test[:,0:pldg_test.shape[1]-5].shape[1], PLDG_CLASS)

DNNDir,DNNDir_history=playcaller_models.PlayTypeDNN(pldg_train[:,0:pldg_train.shape[1]-4],train_dir_class,pldg_train[:,0:pldg_train.shape[1]-4].shape[1], DIR_CLASS)
DNNDir.save(models_path + os.sep + 'play.direction2_DNNPlaytype.h5')
playcaller_models.SaveHistory(DNNDir_history,  history_path + os.sep  + 'play.direction2_DNNPlaytype.hist.txt')
playcaller_models.plotTrainingAcc(DNNDir_history, 'Play Direction - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Accuracy')
playcaller_models.plotTrainingLoss(DNNDir_history, 'Play Direction - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Loss')
playcaller_models.deepPredict(DNNDir, pldg_test[:,0:pldg_test.shape[1]-4], test_dir_class, pldg_test[:,0:pldg_test.shape[1]-4].shape[1], DIR_CLASS)

DNNLen,DNNLen_history=playcaller_models.PlayTypeDNN(pldg_train[:,0:pldg_train.shape[1]-3],train_len_class,pldg_train[:,0:pldg_train.shape[1]-3].shape[1], LEN_CLASS)
DNNLen.save(models_path + os.sep + 'play.length2_DNNPlaytype.h5')
playcaller_models.SaveHistory(DNNLen_history,  history_path + os.sep  + 'play.length2_DNNPlaytype.hist.txt')
playcaller_models.plotTrainingAcc(DNNLen_history, 'Play Length - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Accuracy')
playcaller_models.plotTrainingLoss(DNNLen_history, 'Play Length - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Loss')
playcaller_models.deepPredict(DNNLen, pldg_test[:,0:pldg_test.shape[1]-3], test_len_class, pldg_test[:,0:pldg_test.shape[1]-3].shape[1], LEN_CLASS)

DNNGap,DNNGap_history=playcaller_models.PlayTypeDNN(pldg_train[:,0:pldg_train.shape[1]-3],train_gap_class,pldg_train[:,0:pldg_train.shape[1]-3].shape[1], GAP_CLASS)
DNNGap.save(models_path + os.sep + 'play.gap2_DNNPlaytype.h5')
playcaller_models.SaveHistory(DNNGap_history,  history_path + os.sep  + 'play.gap2_DNNPlaytype.hist.txt')
playcaller_models.plotTrainingAcc(DNNGap_history, 'Play Type - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Accuracy')
playcaller_models.plotTrainingLoss(DNNGap_history, 'Play Type - DNN 2 Hidden Layers, .3 Dropout, Ridge Regression - Loss')
playcaller_models.deepPredict(DNNGap, pldg_test[:,0:pldg_test.shape[1]-3], test_gap_class,  pldg_test[:,0:pldg_test.shape[1]-3].shape[1], GAP_CLASS)
