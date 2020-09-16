# code to tune hyperparameters and get error metrics for 8 ML models using nested cross validation.
# hyperparamaters are tuned on each fold and the best hyperparamaters from the fold are used to get the metrics

import pandas
import scipy
import numpy as np
import datetime
from scipy.stats import zscore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor


# import and define modern data to train model:
filename = '/Users/annaphillips/Google Drive/GLORICH/Si_Env_Data/ModernInputData.csv'
ModernData = pandas.read_csv(filename)
X = zscore(np.array(ModernData.drop(['Stat_Id','ModernLogDSiYield'],axis = 1))) #define independnet variable
Y = np.exp(np.array(ModernData['ModernLogDSiYield'])) #define dependent variable

# manually define random outer outer loops for nested cross-validation
randlist = np.arange(0,6053,1) # list of numbers, same size as data file
np.random.shuffle(randlist) #randomly shuffle list. 

randlist1 = randlist[np.arange(0,1210,1)]
randlist2 = randlist[np.arange(1210,2420,1)]
randlist3 = randlist[np.arange(2420,3630,1)]
randlist4 = randlist[np.arange(3630,4840,1)]
randlist5 = randlist[np.arange(4840,6050,1)] 

Xtrain_out = np.zeros((4840, 30, 5)) 
Xtrain_out[:,:,0] = X[np.concatenate((randlist1, randlist2, randlist3, randlist4)), :]
Xtrain_out[:,:,1] = X[np.concatenate((randlist1, randlist2, randlist3, randlist5)), :]
Xtrain_out[:,:,2] = X[np.concatenate((randlist1, randlist2, randlist4, randlist5)), :]
Xtrain_out[:,:,3] = X[np.concatenate((randlist1, randlist3, randlist4, randlist5)), :]
Xtrain_out[:,:,4] = X[np.concatenate((randlist2, randlist3, randlist4, randlist5)), :]

Ytrain_out = np.zeros((4840,5))
Ytrain_out[:,0] = Y[np.concatenate((randlist1, randlist2, randlist3, randlist4))]
Ytrain_out[:,1] = Y[np.concatenate((randlist1, randlist2, randlist3, randlist5))]
Ytrain_out[:,2] = Y[np.concatenate((randlist1, randlist2, randlist4, randlist5))]
Ytrain_out[:,3] = Y[np.concatenate((randlist1, randlist3, randlist4, randlist5))]
Ytrain_out[:,4] = Y[np.concatenate((randlist2, randlist3, randlist4, randlist5))]

Xtest_out = np.zeros((1210, 30, 5)) 
Xtest_out[:,:,0] = X[randlist5, :]
Xtest_out[:,:,1] = X[randlist4, :]
Xtest_out[:,:,2] = X[randlist3, :]
Xtest_out[:,:,3] = X[randlist2, :]
Xtest_out[:,:,4] = X[randlist1, :]

Ytest_out = np.zeros((1210, 5)) 
Ytest_out[:,0] = Y[randlist5]
Ytest_out[:,1] = Y[randlist4]
Ytest_out[:,2] = Y[randlist3]
Ytest_out[:,3] = Y[randlist2]
Ytest_out[:,4] = Y[randlist1]

# test models

# 1. linear regression:
param_grid_lr = {'fit_intercept': [True]} #parameter grid

R2_results_lr = [] # create empty list for metrics
MAE_results_lr = []
MSE_results_lr = []

for i in range(0,5): # loop through outer splits, use grid search for inner splits
    lr = LinearRegression()
    grid_search = GridSearchCV(estimator = lr, param_grid = param_grid_lr, cv = 5, n_jobs = -1, verbose = 0)
    grid_search.fit(Xtrain_out[:,:,i], Ytrain_out[:,i])
    predictions = grid_search.predict(Xtest_out[:,:,i])
    R2 = r2_score(Ytest_out[:,i], predictions)
    MAE = mean_absolute_error(Ytest_out[:,i], predictions)
    MSE = mean_squared_error(Ytest_out[:,i], predictions)
    R2_results_lr.append(R2) # append metrics to list
    MSE_results_lr.append(MSE)
    MAE_results_lr.append(MAE)

# 2. decision tree
param_grid_dt = {'max_features':[None],
                 'max_leaf_nodes':[None],
                 'min_impurity_decrease':[0.0],
                 'min_impurity_split':[None],
                 'min_samples_leaf':[9,10,11,12,13],
                 'min_samples_split':[3,4]} 

R2_results_dt = [] # create empty list for metrics
MAE_results_dt = []
MSE_results_dt = []

for i in range(0,5): # loop through outer splits, use grid search for inner splits
    dt = DecisionTreeRegressor() 
    grid_search = GridSearchCV(estimator = dt, param_grid = param_grid_dt, cv = 5, n_jobs = -1, verbose = 1)
    grid_search.fit(Xtrain_out[:,:,i], Ytrain_out[:,i])
    predictions = grid_search.predict(Xtest_out[:,:,i])
    R2 = r2_score(Ytest_out[:,i], predictions)
    MAE = mean_absolute_error(Ytest_out[:,i], predictions)
    MSE = mean_squared_error(Ytest_out[:,i], predictions)
    R2_results_dt.append(R2)
    MSE_results_dt.append(MSE)
    MAE_results_dt.append(MAE)
    print('LR best params', grid_search.best_params_) #print the best hyperparameters found by grid search

# 3. K Nearest Neighbour:
param_grid_knn = {'n_neighbors': [5, 6],
                  'weights': ['distance'],
                  'algorithm': ['auto'],
                  'leaf_size': [5]}

R2_results_knn = []
MAE_results_knn = []
MSE_results_knn = []

for i in range(0,5):
    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(estimator = knn, param_grid = param_grid_knn, cv = 5, n_jobs = -1, verbose = 1)
    grid_search.fit(Xtrain_out[:,:,i], Ytrain_out[:,i])
    predictions = grid_search.predict(Xtest_out[:,:,i])
    R2 = r2_score(Ytest_out[:,i], predictions)
    MAE = mean_absolute_error(Ytest_out[:,i], predictions)
    MSE = mean_squared_error(Ytest_out[:,i], predictions)
    R2_results_knn.append(R2)
    MSE_results_knn.append(MSE)
    MAE_results_knn.append(MAE)
    print('KNN best params', grid_search.best_params_)

# 4. Random forest and 5. one step boosted random forest
param_grid_rf = {'bootstrap': [True],
                 'max_depth': [None],
                 'max_features': ['sqrt'],
                 'min_samples_leaf': [1],
                 'min_samples_split': [2, 3],
                 'n_estimators': [500, 600],
                 'oob_score':[True]} # hyperparamater grid for first RF

R2_results_rf = []
MAE_results_rf = []
MSE_results_rf = []

oob_residuals = np.zeros((np.size(Xtrain_out,0),5)) 

param_grid_rf2 = {
    'bootstrap': [True],
    'max_depth': [None],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2],
    #'n_estimators': [100],
    'n_estimators': [200, 300, 400, 500],
    'oob_score':[False]} # hyperparamater grid for second RF

R2_results_osbrf = []
MAE_results_osbrf = []
MSE_results_osbrf = []

for i in range(0,5):
    # initial random forest
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_rf, cv = 5, n_jobs = -1, verbose = 5)
    grid_search.fit(Xtrain_out[:,:,i], Ytrain_out[:,i])
    print('rf1 best params', grid_search.best_params_)
    predictions = grid_search.predict(Xtest_out[:,:,i])
    rf = grid_search.best_estimator_
    rf.fit(Xtrain_out[:,:,i], Ytrain_out[:,i])
    oob_residuals[:,i] = Ytrain_out[:,i] - rf.oob_prediction_
    R2 = r2_score(Ytest_out[:,i], predictions)
    MAE = mean_absolute_error(Ytest_out[:,i], predictions)
    MSE = mean_squared_error(Ytest_out[:,i], predictions)
    R2_results_rf.append(R2)
    MSE_results_rf.append(MSE)
    MAE_results_rf.append(MAE)
    # second random forest
    rf2 = RandomForestRegressor()
    grid_search_rf2 = GridSearchCV(estimator = rf2, param_grid = param_grid_rf2, cv = 5, n_jobs = -1, verbose = 5)
    grid_search_rf2.fit(Xtrain_out[:,:,i], oob_residuals[:,i])
    print('rf2 best params', grid_search_rf2.best_params_)
    # one step boosted random forest result
    osbrf_result = predictions + grid_search_rf2.predict(Xtest_out[:,:,i])
    R2_results_osbrf.append(r2_score(Ytest_out[:,i],osbrf_result))
    MAE_results_osbrf.append(mean_absolute_error(Ytest_out[:,i],osbrf_result))
    MSE_results_osbrf.append(mean_squared_error(Ytest_out[:,i],osbrf_result))     
    
# 6. Support vector regression
param_grid_svr = {'kernel': ['rbf'],
                  'epsilon':[0.15, 0.2],
                  'gamma':['auto'],
                  'tol':[1e-05, 1e-06],
                  'C':[13,14],
                  'max_iter':[1e6]}

R2_results_svr = []
MAE_results_svr = []
MSE_results_svr = []

for i in range(0,5):
    svr = SVR()
    grid_search = GridSearchCV(estimator = svr, param_grid = param_grid_svr, cv = 5, n_jobs = -1, verbose = 5)
    grid_search.fit(Xtrain_out[:,:,i], Ytrain_out[:,i])
    print(grid_search.best_params_)
    predictions = grid_search.predict(Xtest_out[:,:,i])
    R2 = r2_score(Ytest_out[:,i], predictions)
    MAE = mean_absolute_error(Ytest_out[:,i], predictions)
    MSE = mean_squared_error(Ytest_out[:,i], predictions)
    R2_results_svr.append(R2)
    MSE_results_svr.append(MSE)
    MAE_results_svr.append(MAE)

# 7. Gaussian Process Regression
param_grid_gpr = {'alpha':[0.11],
              'kernel': [RationalQuadratic()]}

R2_results_gpr = []
MAE_results_gpr = []
MSE_results_gpr = []


for i in range(0,5):
    gpr = GaussianProcessRegressor()
    grid_search = GridSearchCV(estimator = gpr, param_grid = param_grid_gpr, cv = 5, n_jobs = -1, verbose = 5)
    grid_search.fit(Xtrain_out[:,:,i], Ytrain_out[:,i])
    predictions = grid_search.predict(Xtest_out[:,:,i])
    R2 = r2_score(Ytest_out[:,i], predictions)
    MAE = mean_absolute_error(Ytest_out[:,i], predictions)
    MSE = mean_squared_error(Ytest_out[:,i], predictions)
    R2_results_gpr.append(R2)
    MSE_results_gpr.append(MSE)
    MAE_results_gpr.append(MAE)
    print(grid_search.best_params_)

# 8. Gradient boosted regression
param_grid_gbr = {
    'loss': ['ls'],
    'learning_rate': [0.1],
    'n_estimators': [500],
    'subsample': [1.0],
    'criterion': ['friedman_mse'],
    'min_samples_split':[2, 3],
    'min_samples_leaf': [1, 2],
    'max_depth':[None, 3],
    'min_impurity_decrease': [0],
    'max_features': ['sqrt']}

R2_results_gbr = []
MAE_results_gbr = []
MSE_results_gbr = []

for i in range(0,5):
    gbr = GradientBoostingRegressor()
    grid_search = GridSearchCV(estimator = gbr, param_grid = param_grid_gbr, cv = 5, n_jobs = -1, verbose = 5)
    grid_search.fit(Xtrain_out[:,:,i], Ytrain_out[:,i])
    predictions = grid_search.predict(Xtest_out[:,:,i])
    R2 = r2_score(Ytest_out[:,i], predictions)
    MAE = mean_absolute_error(Ytest_out[:,i], predictions)
    MSE = mean_squared_error(Ytest_out[:,i], predictions)
    R2_results_gbr.append(R2)
    MSE_results_gbr.append(MSE)
    MAE_results_gbr.append(MAE)
    print(grid_search.best_params_)
    
# print results
print('Linear regression')
print('lr R2', R2_results_lr)
print('lr MAE', MAE_results_lr)
print('lr MSE', MSE_results_lr)
print('Decision tree')
print('dt R2', R2_results_dt)
print('dt MAE', MAE_results_dt)
print('dt MSE', MSE_results_dt)
print('knn')
print('knn R2', R2_results_knn)
print('knn MAE', MAE_results_knn)
print('knn MSE', MSE_results_knn)
print('Random Forest')
print('rf R2', R2_results_rf)
print('rf MAE', MAE_results_rf)
print('rf MSE', MSE_results_rf)
print('One step boosted random forest')
print('osbrf R2', R2_results_osbrf)
print('osbrf MAE', MAE_results_osbrf)
print('osbrf MSE', MSE_results_osbrf)
print('Support vector regression')
print('svr R2', R2_results_svr)
print('svr MAE', MAE_results_svr)
print('svr MSE', MSE_results_svr)
print('Gaussian process regression')
print('gpr R2', R2_results_gpr)
print('gpr MAE', MAE_results_gpr)
print('gpr MSE', MSE_results_gpr)
print('Gradient boosted regression')
print('gbr R2', R2_results_gbr)
print('gbr MAE', MAE_results_gbr)
print('gbr MSE', MSE_results_gbr)
