from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Flatten
from keras.utils import np_utils
from keras import losses
from keras import backend as K
from keras.utils.vis_utils import plot_model
import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
import os
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import KFold 
from keras.models import load_model
import cmath 
#give the csv containing complex values here
frame1=pd.read_csv("final_test.csv")
#frame1.drop(frame1.columns[len(frame1.columns)-1], axis=1, inplace=True)
def every(x):	
	try:

		a,b=x.split("+")
		b=b[:-1]
		print(a,b)
		#print(type(a))
		a=float(a)
		b=float(b)
		c=complex(a,b)
		return(abs(c))

	except:
		print("fine")
		return(x)

frame1=frame1.applymap(every)
#csv 
frame1.to_csv('final_test_inter.csv',index=False,header=False)
print("all_done_bro")

train = pd.read_csv("final_test_inter.csv",delimiter=",",header=None)

x = train.values[:,0:train.shape[1]]
carrier = 13
numberWindows = int(x.shape[1]/carrier)
x = x.reshape(x.shape[0] ,carrier,numberWindows, 1)

model1 = Sequential([
	    Conv2D(filters=8, kernel_size=[2,2] , strides=1, activation='relu', input_shape=[carrier,numberWindows,1]),
	    MaxPooling2D(pool_size=[3,2]),
	    Flatten(),
	    Dense(200),				# Nodes in first hidden layer. To add a layers add dense,batchnormalization and acivation.
	    BatchNormalization(),
	    Activation('relu'),
	    Dense(100),				# second hidden layer with 100 nodes
	    BatchNormalization(),
	    Activation('relu'),
	    Dense(3),				# output layer. 2 nodes for binary and number of classes in case of multiclass.
	    BatchNormalization(),
	    Activation('softmax')
	])


	# Model checkpoints and stoppers. Save the best weights and stops the model when no increase in performance to save time.
	#model1.compile(optimizer='nadam', loss='mean_squared_error', metrics=['accuracy'])
	#esc = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto',)
	#cp = ModelCheckpoint(filepath=file, verbose=1, save_best_only=True, save_weights_only=True)


	# fitting the model. 
	#m1 = model1.fit(xTrain, yTrainHot , batch_size=batchSize, epochs=500, callbacks=[esc, cp],validation_data=(xVal,yValHot))


	# loading the best weights
model1.load_weights('model9.hdf5')
modelresults = model1.predict(x)
print (modelresults)
for i2 in range(0,len(modelresults)):
		
    if (modelresults[i2,0] >= modelresults[i2,1]) and (modelresults[i2,1] >= modelresults[i2,2]):
        print ("metal",modelresults[i2,0]*100)
    elif (modelresults[i2,1] >= modelresults[i2,0]) and (modelresults[i2,1] >= modelresults[i2,2]):
        print ("liquid",modelresults[i2,1]*100)
    else:
        print ("no_object",modelresults[i2,2]*100)
 
 

