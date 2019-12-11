# -*- coding: utf-8 -*-
"""
Implementation of EM algorithm for semi-supervised learning
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from keras import utils as np_utils

from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from functools import reduce

from sklearn.utils.extmath import row_norms

from sklearn.preprocessing import StandardScaler

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

#%%
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

#%%
# shuffle all the data, to make sure what ends up in the validation set is random
X_train_labeled, y_train_labeled = shuffle(X_train_labeled, y_train_labeled)

# split the labeled data -> use the training set to estimate the models
# and the validation set to calculate the accuaracy for the models of step (1) and (3)
X_train_labeled, X_valid_labeled = np.split(X_train_labeled, [int(0.9* X_train_labeled.shape[0])], axis=0)
y_train_labeled, y_valid_labeled = np.split(y_train_labeled, [int(0.9* y_train_labeled.shape[0])], axis=0)

# Model #2 : ANN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# --------------------------------------------------------------
# reshape the vectors to one-hot
y_train_labeled = np_utils.to_categorical(y_train_labeled, 10)
y_valid_labeled = np_utils.to_categorical(y_valid_labeled, 10)

# balance
for i in range(y_train_labeled.shape[1]):
    print('No class '+str(i)+ ': '+ str(int(np.sum(y_train_labeled[:, i]))))

# define new data vectors ()
X_train = X_train_labeled.copy()
X_valid = X_valid_labeled.copy()
X_train2 = X_train_unlabeled
X_out = X_test.copy()

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_train2 = scaler.transform(X_train2)
X_out = scaler.transform(X_out)

y_train = y_train_labeled.copy()

# plot to check gaussian distribution of dimensions
fig, ax = plt.subplots(11, 11)
for k in range(121):
    i = int(k/11)
    j = k%11
    ax[i, j].hist(X_train2[:, k])
    
#%% EM Algrithm
XX = np.concatenate((X_train, X_train2), axis = 0)
N = 128 # dimensions
M = 10  # clusters
K = XX.shape[0] # data

# indicator for labeled and unlabeled data -> 1 if labeled
indicator = np.zeros([K])
indicator[:X_train.shape[0]] = 1

# initialization --------------------------------------------------------------
# initizalize weights [M]
w = 1/M *np.ones([M])

# centers [M, dim]
'''
kmeans = KMeans(n_clusters = M, init='k-means++', random_state = 42, max_iter= 10)
kmeans.fit(X_train_labeled)
mu = kmeans.cluster_centers_
'''

# initialize mean
#x_squared_norms = row_norms(X, squared=True)
#KMeans._k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None)

mu = XX[np.random.randint(K, size = M)]

# inizialize covariances
Cov = np.zeros([M, N , N])
for m in range(M):
    Cov[m] = np.eye(N)
    
# inizialize responsibility
gamma = np.zeros([M, K])

# tolerance for convergence
tol = 1e-5

while True:
    # for convergence test -> compare means
    mu_old = mu.copy()
    
    # E-step ------------------------------------------------------------------
    # unnormalized responsibilities (i.e. gamma)
    P = np.zeros([M, K])
    
    # loop over classes
    for m in range(M):
        print('start')
                 
        part1 = 1 / ( ((2* np.pi)**(M/ 2)) * (np.linalg.det(Cov[m])**(1/2)) )
        
        # calculate inverse of covariance matrix for exp( ~(x-mu).T Cov^-1 (x-mu))
        inv_Cov = np.linalg.inv(Cov[m])
         
        # loop over all data points
        for k in range(K):
            
            # term in exponential
            part2 = (-1/2) * ((XX[k]- mu[m]).T.dot(inv_Cov)).dot((XX[k]- mu[m]))
            
            # log(a*b) = log(a)+ log(b) where a is the normalizer and b the exp
            P[m, k] = np.log(part1)+ part2
         
    # normalizer: addition in log domain (don't waste your time here, just
    # check that the columns of gamma sum up to 1 and you can trust this function ;))   
    Z = np.log(np.dot(w.reshape(-1), np.exp(P)))
    
    part0 = np.log(np.dot(w.reshape(-1, 1), np.ones([1, K])))
      
    # normalization: log(P(x, y)/ P(x)) = log(P(x, y))- log(P(x))
    # hard labeling for labeled data
    gamma[:, X_train.shape[0]:] = np.exp(part0+ P- Z)[:, X_train.shape[0]:]
    gamma[:, :X_train.shape[0]] = y_train.T
    
    # np.sum(gamma, axis = 0) to check if columns sum up to 1
            
    # M-step ------------------------------------------------------------------
    # update weights
    w = 1/K * np.sum(gamma, axis = 1)
        
    # loop over classes
    for m in range(M):
        # update mean
        mu[m] = np.dot(gamma[m], XX)/ np.sum(gamma[m])
        
        # subract mean from x [#data pts, dimension]
        x_shift = X_train- mu[m]

        # update covariance
        # this step is very slow!
        cov_full = np.sum([gamma[m, k]* np.dot(x_shift[k].reshape(-1, 1), x_shift[k].reshape(1, -1)) for k in range(K)], axis = 0)/ np.sum(gamma[m])

        # extra step to introduce restrictions on covariance -> here diagonal
        Cov[m] = np.diag(np.diag(cov_full))

    # check convergence
    if np.allclose(mu, mu_old, atol = tol):
        break
    else:
        print('not yet converged')
        
#%%
for m in range(M):
    print('-------------------------')
    print(np.linalg.norm(mu[m]- mu_old[m]))
    print(np.linalg.norm(mu[m]))

#%% Prediction
# unnormalized responsibilities (i.e. gamma)
K = X_valid.shape[0]
p_z_x = np.zeros([M, K])
p_x = np.zeros([K])
P = np.zeros([M, K])

# calculate responsibility i.e. P(z| x, mu, Cov) ------------------------------
# loop over classes
for m in range(M):
    print('start')
    
    part1 = 1 / ( ((2* np.pi)**(M/ 2)) * (np.linalg.det(Cov[m])**(1/2)) )
        
    # calculate inverse of covariance matrix for exp( ~(x-mu).T Cov^-1 (x-mu))
    inv_Cov = np.linalg.inv(Cov[m])
         
    # loop over all data points
    for k in range(K):
            
        # term in exponential
        part2 = (-1/2) * ((X_valid[k]- mu[m]).T.dot(inv_Cov)).dot((X_valid[k]- mu[m]))
            
        # log(a*b) = log(a)+ log(b) where a is the normalizer and b the exp
        P[m, k] = np.log(part1)+ part2
         
# P(x)
p_x = np.dot(w, np.exp(P))

# normalizer: addition in log domain (don't waste your time here, just
# check that the columns of gamma sum up to 1 and you can trust this function ;))   
Z = np.log(np.dot(w.reshape(-1), np.exp(P)))
    
part0 = np.log(np.dot(w.reshape(-1, 1), np.ones([1, K])))
      
# normalization: log(P(x, y)/ P(x)) = log(P(x, y))- log(P(x))
# hard labeling for labeled data
p_z_x = np.exp(part0+ P- Z)

# calculate accuracy ----------------------------------------------------------
accuracy = calc_acc(np.argmax(y_valid_labeled, axis = 1), np.argmax(p_z_x, axis = 0))
print('Accuracy of GM: '+ str(accuracy))

# prediction with conditional probability
# but I think it's also possible to use the joint distribution for the prediction
# which has the advantage of taking outliers into account

#%% Visualization
# plot 2D embedding of training data (labeled + pseudo-labeled)
# colored by respective labelings
from sklearn.decomposition import PCA

cm = plt.get_cmap('tab20', 10)

# dimensionality reduction
pca_kp = PCA(n_components= 2)
pca_kp.fit(XX.T)
        
embeddings = pca_kp.components_
 
# plot clusters with color indicated labels in 2 dimensions
plt.figure()
plt.scatter(embeddings[0, :], embeddings[1, :], c= np.argmax(gamma, axis = 0), cmap = cm)
plt.title('EM')
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')

# dimensionality reduction
pca_kp = PCA(n_components= 3)
pca_kp.fit(XX.T)
        
embeddings = pca_kp.components_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.text2D(0.05, 0.95, 'P = '+ str(p)+ ', K= '+ str(k), transform=ax.transAxes)
ax.scatter(embeddings[0, :], embeddings[1, :], embeddings[2, :], c= np.argmax(gamma, axis = 0), cmap = cm)
ax.set_xlabel('dimension 1')
ax.set_ylabel('dimension 2')
ax.set_zlabel('dimension 3')
plt.show()

