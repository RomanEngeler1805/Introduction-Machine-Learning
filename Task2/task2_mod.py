# -*- coding: utf-8 -*-
"""
Classification by
- Nearest Neighbor (NN)
- Support Vector Machine (SVM)
- Multi Layer Perceptron (MLP)
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors

from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import model_selection

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from sklearn.metrics import precision_score, recall_score, f1_score

from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

seed = 7
np.random.seed(seed)

#%% close plots
plt.close('all')

#%%
# read in the data
df = pd.read_csv("train.csv", sep= ',')
data = df.values
    
id = data[:, 0]
y_read = data[:, 1]
X_read = data[:, 2:18]

# Public set (generate the features)
# training
X_train = X_read[0: 1900]
y_train = y_read[0: 1900]

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# testing i.e. choosing the parameters
X_test = X_read[1900:]
y_test = y_read[1900:]

X_test = scaler.transform(X_test)

# validation i.e. report performance on unused dataset
X_valid = X_read[1800:]
y_valid = y_read[1800:]

X_valid = scaler.transform(X_valid)

# read in the data for submission
df = pd.read_csv("test.csv", sep= ',')
data = df.values
    
index_out = data[:, 0]
X_output = data[:, 1:17]

X_output = scaler.transform(X_output)

X_train_kfold = np.concatenate((X_train,X_test))
y_train_kfold = np.concatenate((y_train,y_test))

X_train_ann = np.concatenate((X_valid,X_train_kfold))
y_train_ann = np.concatenate((y_valid,y_train_kfold))

#%% data analysis
# balance
n_2 = 0
n_1 = 0
n_0 = 0

for i in range(len(X_read)):
    if y_read[i]== 2:
        n_2+= 1
    elif y_read[i]== 1:
        n_1+= 1
    elif y_read[i]== 0:
        n_0+= 1

print('No class 2: ' + str(n_2))
print('No class 1: ' + str(n_1))
print('No class 0: ' + str(n_0))

'''
#%% Nearest neighbors ---------------------------------------------------------------
# array to keep track of accuracy
max_k_number = 10
acc = np.zeros([max_k_number-1])

kf = model_selection.KFold(n_splits=5)


# loop over number of neighbors
for n in range(1, max_k_number):
    # fit model
    nbrs = neighbors.KNeighborsClassifier(n_neighbors=n)
    acc_kf = []
    
    for train_index, test_index in kf.split(X_train_kfold):
        X_train_nn, X_test_nn = X_train_kfold[train_index], X_train_kfold[test_index]
        y_train_nn, y_test_nn = y_train_kfold[train_index], y_train_kfold[test_index]
        nbrs.fit(X_train_nn,y_train_nn)
        # predict class labels
        y_pred = nbrs.predict(X_test_nn)
        # calculate accuracy
        acc_kf.append(accuracy_score(y_test_nn, y_pred))
    
    acc[n-1] = np.mean(acc_kf)
    
# find optimum number of neighbors
n_opt = np.argmin(acc) + 1

# fit model with optimum number of neighbors
nbrs = neighbors.KNeighborsClassifier(n_neighbors=n_opt)
nbrs.fit(X_train_kfold, y_train_kfold)
    
# predict for the unseen validation set
y_pred = nbrs.predict(X_valid)
    
# calculate accuarcy on validation set
accuracy_NN = accuracy_score(y_valid, y_pred)

# print
print('Nearest Neighbors accuracy: '+ str(accuracy_NN)+' with K='+ str(n_opt))


#%% Linear SVM (one-vs-all)
cs = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0]
acc_list_SVM = []

kf = model_selection.KFold(n_splits=5)

# loop over number of neighbors
for c in cs:
    # fit model
    SVM_clf = svm.LinearSVC(C=c)
    acc_kf = []
    
    for train_index, test_index in kf.split(X_train_kfold):
        X_train_nn, X_test_nn = X_train_kfold[train_index], X_train_kfold[test_index]
        y_train_nn, y_test_nn = y_train_kfold[train_index], y_train_kfold[test_index]
        SVM_clf.fit(X_train_nn, y_train_nn)
        # predict class labels
        y_pred = SVM_clf.predict(X_test_nn)
        # calculate accuracy
        acc_kf.append(accuracy_score(y_test_nn, y_pred))
    
    acc_list_SVM.append(np.mean(acc_kf))
    
c_opt = cs[np.argmax(acc_list_SVM)]

SVM_clf = svm.LinearSVC(C=c_opt)
SVM_clf.fit(X_train_kfold, y_train_kfold)

y_pred = SVM_clf.predict(X_valid)

acc_SVM_ova =  accuracy_score(y_valid, y_pred)

print('Linear SVM accuracy (one-vs-all): '+ str(acc_SVM_ova) + ' with C='+str(c_opt))

#%% SVM (one-vs-one)

#first I did a logarithmic grid search like this
cs = [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5]
gammas = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0]
# result was C=1000 and gamma=0.0001


#now converge to more accurate solution
#cs = np.linspace(1e3,4e3,6)
#gammas = np.linspace(5e-5,2e-4,6)
#cs = [2500, 2600, 2700, 2800, 2900]
#gammas = [8.1e-5, 8.15e5, 8.2e-5, 8.25e-5, 8.3e-5]
#One-vs-one SVM accuracy: 0.865 with C=2700 and gamma=8.2e-05
#cs = [2650, 2700, 2750]
#gammas = [8.15e-5, 8.2e-5]

acc_list_SVM = []

kf = model_selection.KFold(n_splits=5)

max_acc = 0
c_opt = -1
gamma_opt = -1

# loop over number of neighbors
for c, gamma in product(cs,gammas):
    # fit model
    SVM_clf = svm.SVC(C=c, gamma=gamma, kernel='rbf')
    acc_kf = []

    for train_index, test_index in kf.split(X_train_kfold):
        X_train_nn, X_test_nn = X_train_kfold[train_index], X_train_kfold[test_index]
        y_train_nn, y_test_nn = y_train_kfold[train_index], y_train_kfold[test_index]
        SVM_clf.fit(X_train_nn, y_train_nn)
        # predict class labels
        y_pred = SVM_clf.predict(X_test_nn)
        # calculate accuracy
        acc_kf.append(accuracy_score(y_test_nn, y_pred))
    
    mean_acc = np.mean(acc_kf)
    
    if mean_acc > max_acc:
        max_acc = mean_acc
        c_opt = c
        gamma_opt = gamma
        #print(max_acc, c_opt, gamma_opt)

SVM_clf = svm.SVC(C=c_opt, kernel='rbf', gamma = gamma_opt)
SVM_clf.fit(X_train_kfold, y_train_kfold)

y_pred = SVM_clf.predict(X_valid)

acc_SVM_ovo =  accuracy_score(y_valid, y_pred)

print('One-vs-one SVM accuracy: '+ str(acc_SVM_ovo) + ' with C='+str(c_opt) + ' and gamma=' +str(gamma_opt))


#%% one-vs-one
# split data sets into classes
train_2 = np.asarray(np.where(y_train== 2))
train_1 = np.asarray(np.where(y_train== 1))
train_0 = np.asarray(np.where(y_train== 0))

train_12 = np.sort(np.concatenate((train_1.T, train_2.T), axis = 0))
train_02 = np.sort(np.concatenate((train_0.T, train_2.T), axis = 0))
train_01 = np.sort(np.concatenate((train_0.T, train_1.T), axis = 0))

# class 2 vs class 1
SVM_clf_12 = svm.SVC(C=c_opt, gamma=gamma_opt, random_state= 42)
SVM_clf_12.fit(X_train[train_12,:].reshape(-1, 16), y_train[train_12].ravel())

y_pred_12 = SVM_clf_12.predict(X_valid)

# class 2 vs class 0
SVM_clf_02 = svm.SVC(C=c_opt, gamma=gamma_opt, random_state= 42)
SVM_clf_02.fit(X_train[train_02].reshape(-1, 16), y_train[train_02].ravel())

y_pred_02 = SVM_clf_02.predict(X_valid)

# class 1 vs class 0
SVM_clf_01 = svm.SVC(C=c_opt, gamma=gamma_opt,random_state= 42)
SVM_clf_01.fit(X_train[train_01].reshape(-1, 16), y_train[train_01].ravel())

y_pred_01 = SVM_clf_01.predict(X_valid)

# evaluate voting scheme

def vote_count(y_pred):
    vote = np.zeros([1, 3])
    
    if y_pred == 0:
        vote[0, 0]+= 1
    elif y_pred == 1:
        vote[0, 1]+= 1
    elif y_pred == 2:
        vote[0, 2]+= 1
        
    return vote

vote = np.zeros([len(y_pred), 3])

# loop
for i in range(len(y_pred)):
    vote1 = vote_count(y_pred_12[i])
    vote2 = vote_count(y_pred_02[i]) 
    vote3 = vote_count(y_pred_01[i]) 
    
    vote[i]= np.add(np.add(vote1, vote2), vote3)
 
# take class with highest rating
y_pred = np.argmax(vote, axis= 1)

# evaluate accuracy
acc_SVM_ovo =  accuracy_score(y_valid, y_pred)


print('One-vs-one SVM accuracy: '+ str(acc_SVM_ovo))
'''

#%% ANN (MLP) ----------------------------------------------------------------

# X_output
def train_NN(no_layer, no_units, activ_fcn, X_train, y_train, X_output):
    # data preprocessing
    y_train2 = np.zeros([y_train.shape[0], 3])
    
    # reshape the vectors to (Nx3)
    for i in range(y_train2.shape[0]):
        if y_train[i] == 0:
            y_train2[i, 0] = 1
        elif y_train[i] == 1:
            y_train2[i, 1] = 1
        elif y_train[i] == 2:
            y_train2[i, 2] = 1    
    
    # NN generation ----------------------------------------------------------
    # Initialize the constructor
    model = Sequential()
    
    # Add an input layer 
    model.add(Dense(16, activation = activ_fcn, input_shape = (16,)))
    #model.add(Dropout(0.2))
    
    # Add hidden layers
    for i in range(0, no_layer):
        model.add(Dense(no_units, activation=activ_fcn))
        #model.add(Dropout(0.2))

    
    # Add an output layer   
    model.add(Dense(3, activation='softmax'))
    

    # Model summary
    model.summary()

    # Train the NN --------------------------------------------------------------
    #learning_rate = 0.12
    #decay_rate = learning_rate/ 80
    #momentum = 0.8
    #sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(loss= 'categorical_crossentropy',
                  optimizer= 'adam',
                  metrics= ['accuracy'])
    
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
                       
    history = model.fit(X_train, y_train2, epochs= 120, batch_size= 25, verbose= 1, validation_split = 0.10, callbacks=callbacks_list, shuffle=True)
    
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    '''
    # Evaluete NN ---------------------------------------------------------------      
    y_pred = model.predict_proba(X_test)
      
    # accuracy
    ac =  accuracy_score(y_test, np.argmax(y_pred, axis= 1))

    # Precision 
    ps= precision_score(y_test, np.argmax(y_pred, axis= 1), average = None)
    # Recall
    rs= recall_score(y_test, np.argmax(y_pred, axis= 1), average = None)
    # F1 score
    fs= f1_score(y_test, np.argmax(y_pred, axis= 1), average = None)
    
    # construct dataframe to monitor performance
    field0= '#layers'
    field1= '#hidden Units'
    field2= 'Activation'
    field3= 'precision'
    field4= 'recall'
    field5= 'f1'
    field6= 'accuracy'

    # field3: [ps], field4: [rs], field5: [fs], 
    d = {field0: [no_layer], field1: [no_units], field2: [activ_fcn], field6: [ac]}
    #df = pd.DataFrame(data=d)
    print(d)
    '''
    y_out = model.predict_proba(X_output)
    
    # output to compare different architectures
    # return np.array([no_layer, no_units, ac])
    
    return np.argmax(y_out, axis= 1)


# parameters -----------------------------------------------------------------------
no_layer= 6
no_units= 6
activ_fcn= 'relu'

'''
# compare different architectures
monitor = np.empty([0, 3])

for i in range(1,2):
    for j in range(16, 17, 4):
        # Train NN
        back = train_NN(i, j, activ_fcn, X_train, y_train, X_test, y_test, X_out)
        monitor= np.append(monitor, back.reshape(1, -1), axis = 0)
'''

y_pred_ANN = train_NN(no_layer, no_units, activ_fcn, X_train, y_train, X_test)

print(accuracy_score(y_test, y_pred_ANN))

y_output = train_NN(no_layer, no_units, activ_fcn, X_train_ann, y_train_ann, X_output)

'''
#%% need to fill the ANN with the weights
model = Sequential()
model.add(Dense(16, activation=activ_fcn, input_shape=(16,)))
model.add(Dense(16, activation=activ_fcn))
model.add(Dense(3, activation='softmax'))
model.summary()
model.load_weights("weights.best.hdf5")
learning_rate = 0.12
decay_rate = learning_rate/ 80
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss= 'categorical_crossentropy',
             	optimizer= 'sgd',
           		metrics= ['accuracy'])
y_output = np.argmax(model.predict_proba(X_out), axis= 1)
'''
data = {'Id' : index_out.astype(int), 'y' : y_output}
df = pd.DataFrame(data)

df.to_csv('output.csv', index=False)
