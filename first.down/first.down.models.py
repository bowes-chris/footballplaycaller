import deeplearning 
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

NB_CLASSES = 2 #number of classes

#load training data from csv file into numpy Array, last column is class label
Train = np.loadtxt('first.down.train.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
#Train_Predictors = Train[:,0:Train.shape[1]-2] # get all rows, all columns except the last column                                               
#Train_class = Train[:,Train.shape[1]-1] # get the last column (the class label)
Train_Predictors = Train[0:1000,0:Train.shape[1]-2] # get all rows, all columns except the last column                                               
Train_class = Train[0:1000,Train.shape[1]-1] # get the last column (the class label)


#load training data from csv file into numpy Array, last column is class label
Test = np.loadtxt('first.down.test.one.hot.csv', delimiter = ',', skiprows = 1) #skip header/label row
Test_Predictors = Test[:,0:Train.shape[1]-2] # get all rows, all columns except the last column                                               
Test_class = Test[:,Train.shape[1]-1] # get the last column (the class label)

Train_class= np_utils.to_categorical(Train_class, NB_CLASSES)
Test_class= np_utils.to_categorical(Test_class, NB_CLASSES)

NUM_PREDICTORS = 136 #number of features



#logistic model
logistic_model, logistic_history = deeplearning.LogisticRegression(Train_Predictors, Train_class, NUM_PREDICTORS, NB_CLASSES)
logistic_model.save('first_down_logistic.h5')
deeplearning.SaveHistory(logistic_history,'first_down_logistic.hist.txt')

#DNN model
DNN_model, DNN_history = deeplearning.DeepNN(Train_Predictors, Train_class, NUM_PREDICTORS, NB_CLASSES)
DNN_model.save('first_down_DNN.h5')
deeplearning.SaveHistory(DNN_history,'first_down_DNN.hist.txt')

DNNDropOut_model,DNNDropOut_history=deeplearning.DeepNNDropOut(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
DNNDropOut_model.save('first_down_DNNDropOut.h5')
deeplearning.SaveHistory(DNNDropOut_history,'first_down_DNNDropOut.hist.txt')

#DNNDOMaxNorm_model,DNNDOMaxNorm_history=deeplearning.DeepNNDropOutMaxNorm(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
#DNNDOMaxNorm_model.save('first_down_DNNDOMaxNorm.h5')
#deeplearning.SaveHistory(DNNDOMaxNorm_history,'first_down_DNNDOMaxNorm.hist.txt')

DNNLasso,DNNLasso_history=deeplearning.DeepNNLasso(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
DNNLasso.save('first_down_DNNLasso.h5')
deeplearning.SaveHistory(DNNLasso_history,'first_down_DNNLasso.hist.txt')

DNNRidge,DNNRidge_history=deeplearning.DeepNNRidge(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
DNNRidge.save('first_down_DNNRidge.h5')
deeplearning.SaveHistory(DNNRidge_history,'first_down_DNNRidge.hist.txt')

DNNElastic,DNNElastic_history=deeplearning.DeepNNElastic(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
DNNElastic.save('first_down_DeepNNElastic.h5')
deeplearning.SaveHistory(DNNElastic_history,'first_down_DNNElastic.hist.txt')

#model8,history8=deeplearning.DeepNNDropOutMaxNormBatchNorm(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES)
#model8.save('mnist_DeepNNDropOutMaxNormBatchNorm.h5')
#deeplearning.SaveHistory(history8,'mnist_DeepNNDropOutMaxNormBatchNorm.hist.txt')

deeplearning.plotTrainingAcc(logistic_history, 'logistic')
deeplearning.plotTrainingAcc(DNN_history, 'DNN')
