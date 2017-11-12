import numpy as np
#simplest model in keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671)



#training hyper-parameters
NB_EPOCH = 10
BATCH_SIZE = 50
VERBOSE = 1 #display results during training
#NB_CLASSES = 2 #number of classes
OPTIMIZER = SGD() # choose optimizer
#OPTIMIZER = keras.optimizers.Adam() # choose optimizer
N_HIDDEN = 128 # number of nodes in the hidden layer
VALIDATION_SPLIT = 0.2 #80% training and 20%validation
METRICS =['accuracy']
LOSS = 'categorical_crossentropy'
DropOut = 0.3

#print(history)
from keras.models import load_model



#input and output layer (no hidden layer)
# here we are not doing k-fold cross validation, which is not usually done for deep learning since it takes forever. the training is split into 80% real training and 20% validation
def LogisticRegression(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):
    model = Sequential()
    #add input + output layer with 784 inputs
    model.add(Dense(NB_CLASSES, input_shape=(NUM_PREDICTORS,)))
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                                validation_split = VALIDATION_SPLIT)
    print(model.summary())
    return model,Tuning

#input, output layer, 1 hidden layer
# here we are not doing k-fold cross validation, which is not usually done for deep learning since it takes forever. the training is split into 80% real training and 20% validation
def DeepNN(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):
    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,)))
    model.add(Activation('relu'))
    #add output layer with NB_CLASSES (10) number of nodes
    model.add(Dense(NB_CLASSES))
    #set activation function for the output layer
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                                validation_split = VALIDATION_SPLIT)
    print(model.summary())
    return model,Tuning

#input, output layer, 1 hidden layer, DropOut on hidden layer
# here we are not doing k-fold cross validation, which is not usually done for deep learning since it takes forever. the training is split into 80% real training and 20% validation
def DeepNNDropOut(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):
    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,)))
    model.add(Activation('relu'))
    model.add(Dropout(DropOut))
    #add output layer with NB_CLASSES (10) number of nodes
    model.add(Dense(NB_CLASSES))
    #set activation function for the output layer
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                                validation_split = VALIDATION_SPLIT)
    print(model.summary())
    return model,Tuning



#input, output layer, 1 hidden layer, DropOut and max-norm constraint on hidden layer
# here we are not doing k-fold cross validation, which is not usually done for deep learning since it takes forever. the training is split into 80% real training and 20% validation
from keras.constraints import maxnorm
def DeepNNDropOutMaxNorm(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):
    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),kernel_constraint = maxnorm(2)))
    model.add(Activation('relu'))
    model.add(Dropout(DropOut))
    #add output layer with NB_CLASSES (10) number of nodes
    model.add(Dense(NB_CLASSES))
    #set activation function for the output layer
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                                validation_split = VALIDATION_SPLIT)
    print(model.summary())
    return model,Tuning

#input, output layer, 1 hidden layer, LASSO regularization on hidden layer
from keras.regularizers import l1
def DeepNNLasso(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):
    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),W_regularizer=l1(0.01), activity_regularizer=l1(0.01)))
    model.add(Activation('relu'))
    #model.add(Dropout(DropOut))
    #add output layer with NB_CLASSES (10) number of nodes
    model.add(Dense(NB_CLASSES))
    #set activation function for the output layer
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                                validation_split = VALIDATION_SPLIT)
    print(model.summary())
    return model,Tuning


#input, output layer, 1 hidden layer, RIDGE regularization on hidden layer
from keras.regularizers import l2
def DeepNNRidge(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):
    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),W_regularizer=l2(0.01), activity_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    #model.add(Dropout(DropOut))
    #add output layer with NB_CLASSES (10) number of nodes
    model.add(Dense(NB_CLASSES))
    #set activation function for the output layer
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                                validation_split = VALIDATION_SPLIT)
    print(model.summary())
    return model,Tuning

#input, output layer, 1 hidden layer, elastic net regularization on hidden layer
from keras.regularizers import L1L2
def DeepNNElastic(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):
     model = Sequential()
     #add hidden layer with NUM_PREDICTORS
     model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),W_regularizer=L1L2(l1=0.01, l2=0.01), activity_regularizer=L1L2(l1=0.01, l2=0.01)))
     model.add(Activation('relu'))
     #model.add(Dropout(DropOut))
     #add output layer with NB_CLASSES (10) number of nodes
     model.add(Dense(NB_CLASSES))
     #set activation function for the output layer
     model.add(Activation('softmax'))
     model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
     Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                                 validation_split = VALIDATION_SPLIT)
     print(model.summary())
     return model,Tuning

#input, output layer, 1 hidden layer, DropOut and max-norm constraint on hidden layer
#here we are not doing k-fold cross validation, which is not usually done for deep learning since it takes forever. the training is split into 80% real training and 20% validation
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
def DeepNNDropOutMaxNormBatchNorm(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):
    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),kernel_constraint = maxnorm(2)))
    model.add(BatchNormalization()) #before activation layer
    model.add(Activation('relu'))
    model.add(Dropout(DropOut))
    #add output layer with NB_CLASSES (10) number of nodes
    model.add(Dense(NB_CLASSES))
    model.add(BatchNormalization()) #before activation layer
    #set activation function for the output layer
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                                validation_split = VALIDATION_SPLIT)
    print(model.summary())
    return model,Tuning


#perform prediction on the test data
def deepPredict(model,Test_Predictors,Test_class,NUM_PREDICTORS, NB_CLASSES):
    score = model.evaluate(Test_Predictors,Test_class)
    print("Test score: ", score[0] )
    print("Test accuracy: ", score[1])

import matplotlib.pyplot as plt
#plot error during training with number of epochs
def plotTrainingLoss(Tuning,Title):
    plt.figure(200)
    plt.plot(Tuning.history['loss'])
    plt.plot(Tuning.history['val_loss'])
    plt.title(Title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'vali'], loc='upper left')
    plt.show()

#plot accuracy during training with number of epochs
def plotTrainingAcc(Tuning, Title):
    plt.figure(100)
    plt.plot(Tuning.history['acc'])
    plt.plot(Tuning.history['val_acc'])
    plt.title(Title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'vali'], loc='upper left')
    plt.show()

#return history of accuracy and loss w.r.t to epoch as a numpy array
def SaveHistory(Tuning,outfile):
    #keys = Tunning.history.keys()
    Hist = np.empty(shape=(len(Tuning.history['val_loss']),4))
    Hist[:,0] = Tuning.history['val_loss']
    Hist[:,1] = Tuning.history['val_acc']
    Hist[:,2] = Tuning.history['loss']
    Hist[:,3] = Tuning.history['acc']
    np.savetxt(outfile, Hist, fmt='%.8f',delimiter=",",header="val_loss,val_acc,train_loss,train_acc",comments="")
    return Hist



#plotTrainingLoss(Tuning)
#plotTrainingAcc(Tuning)
