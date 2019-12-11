# -*- coding: utf-8 -*-
"""
Semi-supervised learning via KNN and MLP
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

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.semi_supervised import label_propagation

#%% close plots
plt.close('all')

#%%
# read in the data ---------------------------------------------
# training data labeled
train_labeled = pd.read_hdf("train_labeled.h5", "train")
y_train_labeled, X_train_labeled = np.split(train_labeled.values, [1], axis=1)

# training data unlabeled
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
X_train_unlabeled = train_unlabeled.values

# test data
test = pd.read_hdf("test.h5", "test")
X_test = test.values
index_out = test.index.values

# function to calcualte accuracy
def calc_acc(y_true, y_pred):
    acc = 0

    # loop over the labels
    for i in range(len(y_true)):
        # if labels coincide, assign point
        if y_true[i] == y_pred[i]:
            acc+= 1

    # normalize by the length
    acc/= len(y_true)

    return acc

# Data Analysis ----------------------------------------------
'''
# Analyse mean and variance in data vector
Mean_X_train = np.zeros([X_train_labeled.shape[1]])
Var_X_train = np.zeros([X_train_labeled.shape[1]])

for i in range(X_train_labeled.shape[1]):
    Mean_X_train[i] = np.mean(X_train_labeled[:, i])
    Var_X_train[i] = np.std(X_train_labeled[:, i])**2

plt.figure()
plt.title('Mean')
plt.plot(Mean_X_train)
plt.xlabel('Dimension')
plt.show()

plt.figure()
plt.title('Variance')
plt.plot(Var_X_train)
plt.xlabel('Dimension')
plt.show()

# scaling the data ---------------------------------------------
#scaler = preprocessing.StandardScaler().fit(X_train_labeled)
#X_train_labeled = scaler.transform(X_train_labeled)
#X_test = scaler.transform(X_test)

'''

# --------------------------------------------------------------
# reshape the vectors to one-hot
y_train_labeled = np_utils.to_categorical(y_train_labeled, 10)

# balance
for i in range(y_train_labeled.shape[1]):
    print('No class '+str(i)+ ': '+ str(np.sum(y_train_labeled[:, i])))

#%% data pre-processing --------------------------------------------
# approach as outlined in "https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/"
# 1) take the labeled data and train a model on it
# 2) take the model to predict the labels for the unlabeled data -> pseudo-labeled data
# 3) use the labeled and the pseudo-labeled data to  train the final model
# 4) take the final model to predict the labels for the test set

# some thoughts:
# (i) should the same model be taken for step (1) and (3)? i.e. twice a SVM
# (ii) linear or non-linear method -> SVM, ANN, kNN, ...
# (iii) scaling of the data? (enable the plots of the mean and variance further down)
# (iv) anothter approach: "http://scikit-learn.org/stable/modules/label_propagation.html"

# shuffle all the data, to make sure what ends up in the validation set is random
X_train_labeled, y_train_labeled = shuffle(X_train_labeled, y_train_labeled)

# split the labeled data -> use the training set to estimate the models
# and the validation set to calculate the accuaracy for the models of step (1) and (3)
X_train_labeled, X_valid_labeled = np.split(X_train_labeled, [int(0.9* X_train_labeled.shape[0])], axis=0)
y_train_labeled, y_valid_labeled = np.split(y_train_labeled, [int(0.9* y_train_labeled.shape[0])], axis=0)


# Model #1 : kNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# can be used as benchmark
# step (1) -----------------------------------------------------------
# initialize kNN and fit
nbrs = KNeighborsClassifier(n_neighbors= 5).fit(X_train_labeled, y_train_labeled.ravel())

# step (2) -----------------------------------------------------------
# predict labels for validation set to calc accuracy
y_valid_predict = nbrs.predict(X_valid_labeled).reshape(-1, 1)
print(calc_acc(y_valid_labeled, y_valid_predict))

# predict labels for unlabeled set -> pseudo-labels
y_train_unlabeled = nbrs.predict(X_train_unlabeled).reshape(-1, 1)

# step (3) -----------------------------------------------------------
X_train = np.concatenate((X_train_labeled, X_train_unlabeled), axis = 0)
y_train = np.concatenate((y_train_labeled, y_train_unlabeled), axis = 0)

# shuffle (possibly not necessary for kNN but most likely necessary for ANN)
X_train, y_train = shuffle(X_train, y_train)

nbrs = KNeighborsClassifier(n_neighbors= 5).fit(X_train, y_train.ravel())

# step (4) -----------------------------------------------------------
# predict labels for validation set to calc accuracy
y_valid_predict = nbrs.predict(X_valid_labeled).reshape(-1, 1)
print(calc_acc(y_valid_labeled, y_valid_predict))

# predict labels for test set
y_output_NN = nbrs.predict(X_test)

# Model #2 : ANN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
adam_decay = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False)

#%% ANN (MLP)
def train_NN(activ_fcn, X_train, y_train, X_valid, y_valid):
    
    # Initialize the constructor
    model = Sequential()
    
    # Add layers
    model.add(Dense(100, activation=activ_fcn, input_shape=(100,)))
    model.add(Dense(100, activation=activ_fcn, kernel_constraint=max_norm(2.)))
    model.add(Dense(10, activation='softmax'))
        
    # Model summary
    model.summary()

    # Train the NN
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_decay,
                  metrics=['accuracy'])
    
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
                       
    history = model.fit(X_train, y_train, epochs = 100, shuffle = True, batch_size = 25,
                        verbose = 1, validation_data = (X_valid, y_valid),
                        callbacks = callbacks_list)

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

# step (1) -----------------------------------------------------------
X_train = X_train_labeled
y_train = np.utils.to_categorical(y_train_labeled, 10)

# train model
train_NN(activ_fcn, X_train, y_train, X_valid_labeled, y_valid_labeled)

# step (2) -----------------------------------------------------------
# need to fill the ANN with the weights
model = Sequential()
model.add(Dense(100, activation=activ_fcn, input_shape=(100,)))
model.add(Dense(100, activation=activ_fcn, kernel_constraint=max_norm(2.)))
model.add(Dense(10, activation='softmax'))

model.load_weights("weights.best.hdf5")
model.compile(loss='categorical_crossentropy',
              optimizer=adam_decay,
              metrics=['accuracy'])

# predict labels for validation set to calc accuracy
y_valid_predict = np.argmax(model.predict(X_valid_labeled), axis=1)
print('Accuracy for ANN = '+ str(calc_acc(y_valid_labeled, y_valid_predict)))

# predict labels for unlabeled set -> pseudo-labels
y_train_unlabeled = np.argmax(model.predict(X_train_unlabeled), axis=1)

# step (3) -----------------------------------------------------------
X_train = np.concatenate((X_train_labeled, X_train_unlabeled), axis = 0)
y_train = np.concatenate((y_train_labeled, y_train_unlabeled), axis = 0)

# shuffle (possibly not necessary for kNN but most likely necessary for ANN)
X_train, y_train = shuffle(X_train, y_train)
y_train = np_utils.to_categorical(y_train_labeled, 10)

# train model
train_NN(activ_fcn, X_train, y_train, X_valid_labeled, y_valid_labeled)

# step (4) -----------------------------------------------------------
# predict labels for validation set to calc accuracy
y_valid_predict = np.argmax(model.predict(X_valid_labeled), axis=1)
print('Accuracy for ANN = '+ str(calc_acc(y_valid_labeled, y_valid_predict)))

# predict labels for test set
y_output_ANN = np.argmax(model.predict(X_test), axis=1)


'''
print(y_train_labeled.shape)

# shuffle all the data, to make sure what ends up in the validation set is random
X_train_labeled, y_train_labeled = shuffle(X_train_labeled, y_train_labeled)

X_train_labeled, X_valid_labeled = np.split(X_train_labeled, [int(0.8*X_train_labeled.shape[0])], axis=0)
y_train_labeled, y_valid_labeled = np.split(y_train_labeled, [int(0.8*y_train_labeled.shape[0])], axis=0)

# to increase weight given to minority classes, 
# y_ints = y_train.flatten()
# class_w = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)

# I tried playing around with the decay a bit, might not be optimal yet
adam_decay = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False)

'''
data = {'Id' : index_out.astype(int), 'y' : y_output_NN}
df = pd.DataFrame(data)
df.to_csv('NN_output.csv', index=False)

data = {'Id' : index_out.astype(int), 'y' : y_output_ANN}
df = pd.DataFrame(data)
df.to_csv('ANN_output.csv', index=False)
