# -*- coding: utf-8 -*-
"""
Semi-supervised learning via PCA + K-Means clustering
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

#%% close plots
plt.close('all')

#%%
# read in the data ---------------------------------------------
train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")


y_train_labeled, X_train_labeled = np.split(train_labeled.values, [1], axis=1)
y_train_unlabeled, X_train_unlabeled = np.split(train_unlabeled.values, [1], axis=1)

X_test = test.values
index_out = test.index.values

# shuffle all the data, to make sure what ends up in the validation set is random
X_train_labeled, y_train_labeled = shuffle(X_train_labeled, y_train_labeled)

X_train_labeled, X_valid_labeled = np.split(X_train_labeled, [int(0.8*X_train_labeled.shape[0])], axis=0)
y_train_labeled, y_valid_labeled = np.split(y_train_labeled, [int(0.8*y_train_labeled.shape[0])], axis=0)


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
'''


# scaling the data
#scaler = preprocessing.StandardScaler().fit(X_train_labeled)
#X_train_labeled = scaler.transform(X_train_labeled)
#X_test = scaler.transform(X_test)


# kmeans --------------------------------------------------------
# want to generate labels for the unlabeled data set to increase
# the training set -> unsupervised learning

# kmeans struggles with high dimensioins -> use PCA to reduce number of dim
# and then apply kmeans to the embedding

acc_PCA = np.zeros([18])
acc_NN = np.zeros([18])
acc_kmeans = 0

# kmeans
kmeans = KMeans(n_clusters = 10, init='k-means++', random_state = 42)
kmeans.fit(X_train_labeled)
y_pred_labeled_kmeans = kmeans.predict(X_valid_labeled)

for i in range(len(y_valid_labeled)):
    if y_valid_labeled[i] == y_pred_labeled_kmeans[i]:
        acc_kmeans += 1

acc_kmeans/= len(y_valid_labeled)
print(acc_kmeans)


for n in range(2, 20):
    # PCA+ kmeans
    # initialize PCA
    PCA_label = PCA(n_components = n)
    # fit training data
    PCA_label.fit(X_train_labeled)
    # extract embeddings
    PCA_train = PCA_label.components_
    # predict embeddings for validation set
    PCA_valid = PCA_label.transform(X_valid_labeled)

    # initialize kmeans clustering
    kmeans_pca = KMeans(n_clusters = 10, init='k-means++', random_state = 42)
    # fit kmeans clsutering
    kmeans_pca.fit(PCA_train.T)
    # predict labels for validation set
    y_pred_labeled_PCA = kmeans_pca.predict(PCA_valid)

    # kNN
    # initialize kNN and fit
    nbrs = KNeighborsClassifier(n_neighbors= n).fit(X_train_labeled, y_train_labeled.ravel())
    # predict labels for validation set
    y_pred_labeled_NN = nbrs.predict(X_valid_labeled)

    # calculate accuracy
    for i in range(len(y_valid_labeled)):
        if y_valid_labeled[i] == y_pred_labeled_PCA[i]:
            acc_PCA[n- 2] += 1

        if y_valid_labeled[i] == y_pred_labeled_NN[i]:
            acc_NN[n- 2] += 1

    acc_PCA[n- 2]/= len(y_valid_labeled)
    acc_NN[n- 2]/= len(y_valid_labeled)

# output
print(acc_kmeans)
print(acc_PCA)
print(acc_NN)



# ----------------------------------------------------------------------
# it was seen that PCA with kmeans performs the best

# track accuracy
acc_PCA = np.zeros([18, 10])
# track cluster centers
kmeans_clusters = np.zeros([18, 10], dtype= object)

for n in range(2, 20):
    # PCA+ kmeans
    PCA_label = PCA(n_components = n)
    PCA_label.fit(X_train_labeled)
    PCA_train = PCA_label.components_

    PCA_valid = PCA_label.transform(X_valid_labeled)

    for k in range(acc_PCA.shape[1]):
        # initialize kmeans
        kmeans_pca = KMeans(n_clusters = 10, init='k-means++')
        # fit kmeans
        kmeans_pca.fit(PCA_train.T)
        # save cluster centers
        kmeans_clusters[n-2, k] = kmeans_pca.cluster_centers_ 
        # predict labels for validation set
        y_pred_labeled_PCA = kmeans_pca.predict(PCA_valid)

        # compare predicted with true labels
        for i in range(len(y_valid_labeled)):
            if y_valid_labeled[i] == y_pred_labeled_PCA[i]:
                acc_PCA[n- 2, k] += 1

        acc_PCA[n- 2, k]/= len(y_valid_labeled)

# plot accuracy
plt.figure()
for n in range(2, 20):
    plt.plot(acc_PCA[n-2, :])
    plt.legend('No of components = '+ str(n))
plt.xlabel('Realisation')
plt.show()

kmeans_id_max = np.where(acc_PCA == np.max(acc_PCA))

print(kmeans_id_max[:])
print(kmeans_id_max[1][0])

kmeans_clusters_max = kmeans_clusters[kmeans_id_max[0][0], kmeans_id_max[1][0]]

PCA_label = PCA(n_components = kmeans_id_max[0][0]+ 2)
PCA_label.fit(X_train_labeled)
PCA_train = PCA_label.components_

PCA_valid = PCA_label.transform(X_test)

# initialize kmeans
kmeans_pca = KMeans(n_clusters = 10, init = kmeans_clusters_max, n_init = 10)
        # fit kmeans
kmeans_pca.fit(PCA_train.T)
# predict labels for validation set
y_output = kmeans_pca.predict(PCA_valid)


data = {'Id' : index_out.astype(int), 'y' : y_output}
df = pd.DataFrame(data)

df.to_csv('PCA_Kmeans_output.csv', index=False)
