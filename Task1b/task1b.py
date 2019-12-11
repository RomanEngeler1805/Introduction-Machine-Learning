# -*- coding: utf-8 -*-
"""
Script for linear regression (LASSO, Ridge)
"""
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as sklLM

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold

#%% close plots
plt.close('all')

#%%
# read in the data
df = pd.read_csv("train.csv", sep= ',')
data = df.values
    
id = data[:, 0]
y_read = data[:, 1]
X_read = data[:, 2:7]

# Public set (generate the features)
X_temp = X_read[0: 700]

X_quad = np.square(X_temp)
X_exp =  np.exp(X_temp)
X_cos = np.cos(X_temp)

X = np.concatenate((X_temp, X_quad, X_exp, X_cos, np.ones([len(X_temp), 1])), axis=1)

y = y_read[0:700]   

# Private set (mimic online submission)
X_temp2 = X_read[700: 900]

X_private = np.concatenate((X_temp2, np.square(X_temp2), np.exp(X_temp2), np.cos(X_temp2), np.ones([len(X_temp2), 1])), axis=1)

y_private = y_read[700:900]

plt.figure()
plt.plot(y_read)
plt.show()


#%%
# Compute RMSE using N-fold x-validation
k_folds= 100
kf = KFold(len(X), n_folds=k_folds)
alpha= [0.04, 0.1, 0.15, 0.6, 10.0, 14.0, 18.0, 22.0, 50, 200.0, 1200.0, 4000, 1E5]

# arrays to store RMSE (Lasso, Ridge regression, Analytical Equation for Ridge Regression)
# err: [# alphas, 1]
rmse_cv_lasso = np.zeros([len(alpha), 1])
rmse_cv_ridge = np.zeros([len(alpha), 1])
rmse_cv_ana = np.zeros([len(alpha), 1])

rmse_private = np.zeros([len(alpha), 1])

# vector to store model parameters
# w: [#alphas, #params]
w_ridge = np.zeros([len(alpha), len(X.T)])
w_lasso = np.zeros([len(alpha), len(X.T)])
w_ana = np.zeros([len(alpha), len(X.T)])

# detailed monitoring of variation of solution vector for each fold
# for insights about stability of chosen model
# var: [#alphas, #params]
var_w_ridge =  np.zeros([len(alpha), len(X.T)])
var_w_lasso =  np.zeros([len(alpha), len(X.T)])
var_w_ana =  np.zeros([len(alpha), len(X.T)])

# loop over all alpha values
for i in range(len(alpha)):
    # initialize error
    xval_err_ridge = 0
    xval_err_lasso = 0
    xval_err_ana = 0
    
    # initialize model
    ridge = sklLM.Ridge(fit_intercept=False, alpha= alpha[i], normalize= False)
    lasso = sklLM.Lasso(fit_intercept=False, alpha= alpha[i], normalize= False)
    
    # loop over training/ test sets
    for train,test in kf:
        # fit RIDGE model and predict -----------------------------------------
        ridge.fit(X[train], y[train])
        p_r = ridge.predict(X[test])
        
        # calculate RMSE
        xval_err_ridge += mean_squared_error(p_r, y[test])** 0.5
        
        # avaerage solution vector
        w_ridge[i]= w_ridge[i]+ ridge.coef_/ k_folds
        
        # calculate variance
        var_w_ridge[i] = var_w_ridge[i]+ ridge.coef_**2
        
        
        # fit LASSO model and predict -----------------------------------------
        lasso.fit(X[train], y[train])
        p_l = lasso.predict(X[test])
        
        # calculate RMSE
        e_l = p_l - y[test]
        xval_err_lasso += mean_squared_error(p_l, y[test])** 0.5
        
        # avaerage solution vector
        w_lasso[i]= w_lasso[i]+ lasso.coef_/ k_folds
        
        # calculate variance
        var_w_lasso[i] = var_w_lasso[i]+ lasso.coef_**2


        # analytical solution to ridge regression -----------------------------
        w = np.linalg.solve(np.dot(X[train].T, X[train])+
                            alpha[i]* np.identity(X[train].shape[1]),
                            np.dot(X[train].T, y[train]))
        y_pred = np.dot(X[test], w.reshape(-1, 1))
        
        # calculate RMSE
        xval_err_ana += mean_squared_error(y_pred, y[test].reshape(-1, 1))** 0.5
        
        # average solution vector
        w_ana[i] = w_ana[i]+ w/ k_folds
        
        # variance
        var_w_ana[i] = var_w_ana[i]+ w**2

        
    # Var ~ E[x^2] - E[x]^2 ---------------------------------------------------
    var_w_ridge[i] = 1/ k_folds* (var_w_ridge[i]- k_folds* w_ridge[i]**2)
    var_w_lasso[i] = 1/ k_folds* (var_w_lasso[i]- k_folds* w_lasso[i]**2)
    var_w_ana[i] = 1/ k_folds* (var_w_ana[i]- k_folds* w_ana[i]**2)
       
    # RMSE calculation
    rmse_cv_ridge[i] = xval_err_ridge/ k_folds
    rmse_cv_lasso[i] = xval_err_lasso/ k_folds
    rmse_cv_ana[i] = xval_err_ana/ k_folds
    
    # RMSE for private set -----------------------------------------------------
    y_pred = np.dot(X_private, w_lasso[i]) # change here to lasso XXXX and observe
    rmse_private[i] = mean_squared_error(y_pred, y_private)** 0.5
    
    # Plot the residuals (to check gaussian)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(y_private, c= 'b')
    ax[0].plot(y_pred, c= 'r')
    ax[0].set_title(alpha[i])

    ax[1].plot((y_private- y_pred), c= 'b')
    ax[1].set_title('Residuals'+str(alpha[i]))
    
    ax[2].hist((y_private- y_pred), bins='auto')  # arguments are passed to np.histogram
    ax[2].set_title("Histogram with 'auto' bins")
    
#%% Plot
fig, ax = plt.subplots(3, 3, figsize=(15,5))

# RMSE error vs alphas
ax[0, 0].plot(alpha, rmse_cv_ridge)
ax[0, 0].set_xscale('log')
ax[0, 0].set_title('RMSE Ridge')
ax[0, 1].plot(alpha, rmse_cv_lasso)
ax[0, 1].set_xscale('log')
ax[0, 1].set_title('RMSE Lasso')
ax[0, 2].plot(alpha, rmse_cv_ana)
ax[0, 2].set_xscale('log')
ax[0, 2].set_title('RMSE Ridge Analytical')

# Variance of coefficients vs alphas
ax[1, 0].plot(alpha, w_ridge)
ax[1, 0].set_xscale('log')
ax[1, 0].set_title('Parameters Ridge')
ax[1, 1].plot(alpha, w_lasso)
ax[1, 1].set_xscale('log')
ax[1, 1].set_title('Parameters Lasso')
ax[1, 2].plot(alpha, w_ana)
ax[1, 2].set_xscale('log')
ax[1, 2].set_title('Parameters Ridge Analytical')

# Variance of coefficients vs alphas
ax[2, 0].plot(alpha, var_w_ridge)
ax[2, 0].set_xscale('log')
ax[2, 0].set_title('Var(Params) Ridge')
ax[2, 1].plot(alpha, var_w_lasso)
ax[2, 1].set_xscale('log')
ax[2, 1].set_title('Var(Params) Lasso')
ax[2, 2].plot(alpha, var_w_ana)
ax[2, 2].set_xscale('log')
ax[2, 2].set_title('Var(Params) Ridge Analytical')

fig.show()        

# Plot RMSE vs alpha on private set
plt.figure()
plt.plot(alpha, rmse_private)
plt.xscale('log')
plt.title('RMSE Private (Submission)')
plt.show()
    
# Visualize data dependency
fig, ax = plt.subplots(4, 5, figsize = (3, 8))
for i in range(X.shape[1]- 1):
    xx = i% 4
    yy = int(i/4)
    ax[xx, yy].scatter(X[:, i], y)
    ax[xx, yy].scatter(X[:, i], w_ridge[7, i]* X[:, i], c= 'r')
    
fig.suptitle('Data fitting with Alpha = 14')


#%% output for quick check
ind_R= np.argmin(rmse_cv_ridge)
print('Optimal lambda for Ridge: '+str(int(alpha[ind_R])))

ind_L= np.argmin(rmse_cv_lasso)
print('Optimal lambda for Lasso: '+str(int(alpha[ind_L])))

ind_A= np.argmin(rmse_cv_ana)
print('Optimal lambda for Ridge (analytical): '+str(int(alpha[ind_A])))

#%% write solution file
df = pd.DataFrame(data = w_ridge[ind_R])
df.to_csv('output_Ridge_lam'+str(int(alpha[ind_R]))+'_kfold'+str(k_folds)+'.csv', header= None, index= None)

df2 = pd.DataFrame(data = w_lasso[3])
df2.to_csv('output_Lasso_lam'+str(int(alpha[3]))+'_kfold'+str(k_folds)+'.csv', header= None, index= None)

df3 = pd.DataFrame(data = w_ana[ind_A])
df3.to_csv('output_Analytical_lam'+str(int(alpha[ind_A]))+'_kfold'+str(k_folds)+'.csv', header= None, index= None)
