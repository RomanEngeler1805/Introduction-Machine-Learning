# -*- coding: utf-8 -*-
"""
Support Vector Machine (SVM) for multi-class classification
"""

#sets operating folder where the train and test files are located
'''
import os
cwd = os.getcwd()
os.chdir(r'C:\Users\Edoardo\Google Drive\Uni Edo\Master\Semester 2\Machine learning\Project\task2_s82hdj')
'''

import pandas as pd
import numpy as np
#np.set_printoptions(threshold=np.nan)
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score
#from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# reading in data
train_df = pd.read_csv('train.csv')
X_complete = train_df.drop(columns=['Id','y']).values
y_complete = train_df['y'].values

test_df = pd.read_csv('test.csv')
X_test = test_df.drop(columns=['Id']).values

# Need for standardization
for i in range(X_complete.shape[1]):
    Xmean = np.mean(X_complete[:, i])
    Xstd = np.std(X_complete[:, i])
    print('Column '+ str(i)+ ', mean= '+ str(Xmean)+ ', stdv= '+ str(Xstd))


# Standardization
scaler = MinMaxScaler().fit(X_complete)
X_complete = scaler.transform(X_complete)
X_test = scaler.transform(X_test)


#Split train set to obtain validation set
X_validation = X_complete[1700:2000, :]
y_validation = y_complete[1700:2000]
X = X_complete[0:1700, :]
y = y_complete[0:1700]


# Model properties
# one-vs-one (ovo) or ove-vs-rest (ovr)
mode = 'ovo' # 'ovo' vs 'ovr'
kkkernel = 'poly'  #'poly', 'rbf', 'linear'
# degree of polynomial kernel (will be ignored for others)
if kkkernel == 'poly':
    dddegree = 4
else:
    dddegree = 2                                            
# kernel coefficient  (e.g. for RBF exp(-||x-x'||/gamma²))
ggamma = np.array([1E-1, 1E0, 2, 5])

accuracy_kf_scores = np.empty([0, 4])
c_list = np.linspace(1, 9, 5)#[0.5, 0.75, 1, 1.25, 1.5, 1.75] #0.01, 0.1, 0.25,  #0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.2,5 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5
# c_list = [5]

# loop over gamma (kernel coefficient e.g. for RBF exp(-||x-x'||/gamma²))
for g in ggamma:
    # loop over degree of polynomial (ignored by other kernels)
    for d in range(1, dddegree):
        # loop ver c/ lambda (penalty for misclassification)
        for c in c_list:
            # splitting for corss-validation
            kf = model_selection.KFold(n_splits=5, shuffle=True).split(X)
            # vector to store model properties and achieved accuracy
            accList = []
            # initialize SVM
            classifier = svm.SVC(C=c, kernel=kkkernel, degree=d, gamma = g, decision_function_shape=mode)

            # cross-validation
            for trainIndex, testIndex in kf:
                X_train_kf, X_test_kf = X[trainIndex], X[testIndex]
                Y_train_kf, Y_test_kf = y[trainIndex], y[testIndex]

                # fit model
                classifier.fit(X_train_kf, Y_train_kf)

                # predict
                y_pred_kf = classifier.predict(X_test_kf)

                # accuracy
                acc = accuracy_score(Y_test_kf, y_pred_kf)
                accList.append(acc)

            # output to user to see progress and accuracy
            print('degree of poly: '+ str(d)+ ', c: '+ str(c)+ ', gamma : '+ str(g)+ ', accuracy of folds = '+ str(np.mean(accList)))

            # store model properties and accuracy
            new_score = np.array([d, c, g, np.mean(accList)]).reshape(1, -1)
            accuracy_kf_scores = np.append(accuracy_kf_scores, new_score, axis = 0)


# extract best performing parameters
maxIndex = np.argmax(accuracy_kf_scores[:, 3])
optimum_c = accuracy_kf_scores[maxIndex, 1]
optimum_deg = accuracy_kf_scores[maxIndex, 0]
optimum_gam = accuracy_kf_scores[maxIndex, 2]

# print('Accuracy kf scores for each c = ', accuracy_kf_scores)
print('Max accuracy from KF test = ', accuracy_kf_scores[maxIndex, 3])
print('Optimum degree = ' + str(optimum_deg))
print('Optimum c = ' + str(optimum_c))
print('Optimum gamma = ' + str(optimum_gam))

#evaluate accuracy on accuracy set
classifier = svm.SVC(C=optimum_c, kernel=kkkernel, degree=optimum_deg, gamma = optimum_gam, decision_function_shape=mode)
classifier.fit(X,y)
y_validation_pred = classifier.predict(X_validation)
print('Accuracy on validation set = '+ str(accuracy_score(y_validation, y_validation_pred)))


#fitting classifier on whole data and predicting
classifier.fit(X_complete,y_complete)
y_pred = classifier.predict(X_test)

#write to file
indexArray = np.arange(2000, 5000, 1)
dfPreparation = {'Id' : indexArray, 'y' : y_pred} 
y_pred_df = pd.DataFrame(dfPreparation)
y_pred_df.to_csv('result_SVM_'+ kkkernel+ '_'+ mode + '_c_'+ str(optimum_c)+ '_d_'+ str(optimum_deg)+ '_g_'+ str(optimum_gam)+ '.csv', index=False)
