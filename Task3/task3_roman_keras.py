# -*- coding: utf-8 -*-
"""
Multi Layer Perceptron (MLP) for classification with class imbalance

"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import shuffle, class_weight
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras import utils as np_utils
from keras.constraints import max_norm
from keras import optimizers

#%% close plots
plt.close('all')

#%%
# read in the data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

y_train, X_train = np.split(train.values, [1], axis=1)

X_test = test.values
index_out = test.index.values

# shuffle all the data, to make sure what ends up in the validation set is random
X_train, y_train = shuffle(X_train, y_train)

X_train, X_valid = np.split(X_train, [int(0.8*X_train.shape[0])], axis=0)
y_train, y_valid = np.split(y_train, [int(0.8*y_train.shape[0])], axis=0)

# scaling the data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


#%% data analysis
# balance
n_4 = 0
n_3 = 0
n_2 = 0
n_1 = 0
n_0 = 0

for i in range(X_train.shape[0]):
    if y_train[i] == 4:
        n_4 += 1
    elif y_train[i] == 3:
        n_3 += 1
    elif y_train[i] == 2:
        n_2 += 1
    elif y_train[i] == 1:
        n_1 += 1
    elif y_train[i] == 0:
        n_0 += 1

print('No class 4: ' + str(n_4))
print('No class 3: ' + str(n_3))
print('No class 2: ' + str(n_2))
print('No class 1: ' + str(n_1))
print('No class 0: ' + str(n_0))

#%%
# to increase weight given to minority classes, 
y_ints = y_train.flatten()
class_w = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)

# I tried playing around with the decay a bit, might not be optimal yet
adam_decay = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False)

# reshape the vectors to one-hot
y_train = np_utils.to_categorical(y_train, 5)
y_valid = np_utils.to_categorical(y_valid, 5)

#%% ANN (MLP)
def train_NN(activ_fcn, X_train, y_train):
    
    # Initialize the constructor
    model = Sequential()
    
    # Add layers
    model.add(Dense(100, activation=activ_fcn, input_shape=(100,)))
    model.add(Dropout(0.20))
    model.add(Dense(100, activation=activ_fcn, kernel_constraint=max_norm(2.)))
    model.add(Dropout(0.30))
    model.add(Dense(100, activation=activ_fcn, kernel_constraint=max_norm(2.)))
    model.add(Dropout(0.30))
    model.add(Dense(5, activation='softmax'))
    
    # Model summary
    model.summary()

    # Train the NN
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_decay,
                  metrics=['accuracy'])
    
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
                       
    history = model.fit(X_train, y_train, epochs=140, shuffle=True, batch_size=25, verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks_list, class_weight=class_w)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

activ_fcn = 'relu'
train_NN(activ_fcn, X_train, y_train)


#%% need to fill the ANN with the weights
model = Sequential()
model.add(Dense(100, activation=activ_fcn, input_shape=(100,)))
model.add(Dropout(0.2))
model.add(Dense(100, activation=activ_fcn, kernel_constraint=max_norm(2.)))
model.add(Dropout(0.30))
model.add(Dense(100, activation=activ_fcn, kernel_constraint=max_norm(2.)))
model.add(Dropout(0.30))
model.add(Dense(5, activation='softmax'))

model.load_weights("weights.best.hdf5")
model.compile(loss='categorical_crossentropy',
              optimizer=adam_decay,
              metrics=['accuracy'])
y_output = np.argmax(model.predict(X_test), axis=1)

data = {'Id' : index_out.astype(int), 'y' : y_output}
df = pd.DataFrame(data)

df.to_csv('NN_output.csv', index=False)
