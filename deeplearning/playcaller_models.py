import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
import os
np.random.seed(1671)

from keras.models import load_model
from keras.constraints import maxnorm
from keras.regularizers import l1
from keras.regularizers import l2
from keras.regularizers import L1L2
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint 
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=3)

def PlayTypeDNN(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):

    #training hyper-parameters
    NB_EPOCH = 500
    BATCH_SIZE = 50
    VERBOSE = 1 #display results during training
    OPTIMIZER = SGD() # choose optimizer
    #OPTIMIZER = Adam() # choose optimizer
    VALIDATION_SPLIT = 0.2 #80% training and 20%validation
    METRICS =['accuracy']
    LOSS = 'categorical_crossentropy'
    DROP_OUT = 0.3
    N_HIDDEN = 100 # number of nodes in the hidden layer

    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    #add output layer with NB_CLASSES 
    model.add(Dense(NB_CLASSES))    
    #set activation function for the output layer
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    filepath="model1_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]    
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                       validation_split = VALIDATION_SPLIT,callbacks=[early_stopping_monitor])
    print(model.summary())
    return model,Tuning

def FirstDownDNN(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):

    #training hyper-parameters
    NB_EPOCH = 500
    BATCH_SIZE = 50
    VERBOSE = 1 #display results during training
    #OPTIMIZER = SGD() # choose optimizer
    OPTIMIZER = Adam() # choose optimizer
    VALIDATION_SPLIT = 0.2 #80% training and 20%validation
    METRICS =['accuracy']
    LOSS = 'categorical_crossentropy'
    DROP_OUT = 0.3
    N_HIDDEN = 100 # number of nodes in the hidden layer

    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    #add output layer with NB_CLASSES 
    model.add(Dense(NB_CLASSES))    
    #set activation function for the output layer
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    filepath="model1_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]    
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                       validation_split = VALIDATION_SPLIT,callbacks=[early_stopping_monitor])
    print(model.summary())
    return model,Tuning

def TouchdownDNN(Train_Predictors,Train_class,NUM_PREDICTORS, NB_CLASSES):

    #training hyper-parameters
    NB_EPOCH = 500
    BATCH_SIZE = 50
    VERBOSE = 1 #display results during training
    #OPTIMIZER = SGD() # choose optimizer
    OPTIMIZER = Adam() # choose optimizer
    VALIDATION_SPLIT = 0.2 #80% training and 20%validation
    METRICS =['accuracy']
    LOSS = 'categorical_crossentropy'
    DROP_OUT = 0.3
    N_HIDDEN = 100 # number of nodes in the hidden layer

    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    #add output layer with NB_CLASSES 
    model.add(Dense(NB_CLASSES))    
    #set activation function for the output layer
    model.add(Activation('softmax'))
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    filepath="model1_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]    
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                       validation_split = VALIDATION_SPLIT,callbacks=[early_stopping_monitor])
    print(model.summary())
    return model,Tuning

def YardsGainedDNN(Train_Predictors,Train_class,NUM_PREDICTORS):

    #training hyper-parameters
    NB_EPOCH = 100
    BATCH_SIZE = 25
    VERBOSE = 1 #display results during training
    #OPTIMIZER = SGD() # choose optimizer
    OPTIMIZER = Adam() # choose optimizer
    VALIDATION_SPLIT = 0.2 #80% training and 20%validation
    METRICS =['accuracy']
    LOSS = 'mean_squared_error'
    DROP_OUT = 0.3
    N_HIDDEN = 100 # number of nodes in the hidden layer

    model = Sequential()
    #add hidden layer with NUM_PREDICTORS
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    model.add(Dense(N_HIDDEN, input_shape=(NUM_PREDICTORS,),
                    #W_regularizer=l2(0.01), activity_regularizer=l2(0.01))) #Ridge
                    W_regularizer=l1(0.01), activity_regularizer=l1(0.01))) #Lasso
    model.add(Activation('relu'))
    model.add(Dropout(DROP_OUT))
    #add output layer with NB_CLASSES 
    model.add(Dense(1))    
    #set activation function for the output layer
    model.compile(loss=LOSS, optimizer = OPTIMIZER, metrics =METRICS)
    filepath="model1_dropout_"+str(DROP_OUT)+"_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]    
    Tuning = model.fit(Train_Predictors,Train_class,batch_size=BATCH_SIZE, epochs = NB_EPOCH, verbose = VERBOSE,
                       validation_split = VALIDATION_SPLIT,callbacks=[early_stopping_monitor])
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