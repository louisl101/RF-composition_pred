# %%# -*- coding:utf-8 -*-
# louis1001
import sklearn
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, KFold
import math
import pandas as pd
import numpy as np
import scipy.stats
from math import sqrt
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import time
from tqdm import trange
from joblib import dump, load
import pickle
import itertools

# %%
def RMSE(obs, pred):
    return sqrt(mean_squared_error(obs, pred))

def NRMSE(obs, pred):
    return RMSE(obs, pred) / (max(obs) - min(obs))

# %%
base=pd.read_csv('species_composition_pred/data/mjosa_data_relative_2month.csv')
# base=pd.read_csv('species_composition_pred/data/mjosa_data.csv')
# base=base.loc[lambda x: x['year']>=2000].reset_index(drop=True).fillna(0)

# absolute
# wq=base.iloc[:,15:20].columns.to_numpy()
# metro_hydro=base.iloc[:,20:55].columns.to_numpy()
# spa=base.iloc[:,56:99].columns.to_numpy()
# algae=base.iloc[:,6:12].columns.to_numpy()
# # imp=pd.read_csv('species_composition_pred/data/imp_all.csv')
# base[algae]=np.log(base[algae]+1)

##
# # relative
wq=base.iloc[:,20:25].columns.to_numpy()
metro_hydro=base.iloc[:,24:60].columns.to_numpy()
spa=base.iloc[:,61:99].columns.to_numpy()
algae=base.iloc[:,6:20].columns.to_numpy()
imp=pd.read_csv('species_composition_pred/data/imp_RCB_all.csv')

algae=np.array(['Phytoplankton'])
base['Phytoplankton']=np.log10(base['Phytoplankton'])
## 2 month
# # relative
wq=base.iloc[:,20:25].columns.to_numpy()
metro_hydro=base.iloc[:,28:99].columns.to_numpy()
spa=base.iloc[:,26:28].columns.to_numpy()
algae=base.iloc[:,6:20].columns.to_numpy()
imp=pd.read_csv('species_composition_pred/data/imp_RCB_all.csv')

# algae=np.array(['Phytoplankton'])
base['Phytoplankton']=np.log10(base['Phytoplankton'])
# %%   model Hyper_params
Hyper=dict(
	n_estimators=[100,250,500,1000,1500],
	max_features=[0.1,0.3,0.5,0.7,0.85,0.99]
)

Hyper_params=[v for v in Hyper.values()]
# %%   model fitting
optim_keeper=pd.DataFrame()
for M in range(0,1):
    M=40
    base_clib, base_validate= train_test_split(base, test_size=0.2, random_state=40)
    base_clib=base_clib.reset_index(drop=True).fillna(0)
    base_validate=base_validate.reset_index(drop=True).fillna(0)
    normaler=StandardScaler()
    ## create containers to record each outcomes
    ## training
    for j in algae:
        print(j)
        ### all
        env=np.concatenate([wq,metro_hydro])
        all_feature=np.concatenate([env,spa])
        X_clib=base_clib[all_feature]
        X_validate=base_validate[all_feature]
        ### selected
        # imp_x=imp.loc[lambda x: x['algae']==j].reset_index(drop=True)
        # env=np.concatenate((wq,imp_x['variable']))
        # important_feature=np.concatenate((env,spa))
        # X_clib=base_clib[important_feature]
        # X_validate=base_validate[important_feature]
        ###
        Y_clib = base_clib[algae][j]
        Y_validate = base_validate[algae][j]
        cv = KFold(n_splits=10,shuffle=True,random_state=999)
        counts=0
        for n_estimators,max_features in itertools.product(*Hyper_params):
            cv_results = pd.DataFrame(index=range(0, 999),
                                 columns=['train_r2', 'train_rmse', 'train_mse',
                                          'test_r2', 'test_rmse', 'test_mse',
                                          'clib_r2', 'clib_rmse', 'clib_mse',
                                          'pred_r2', 'pred_rmse', 'pred_mse',
                                          'n_estimators','max_features','algae'
                                          ])
            i=0
            for train_index, test_index in cv.split(X_clib):
                X_train, X_test = X_clib.loc[train_index], X_clib.loc[test_index]
                Y_train, Y_test = Y_clib.loc[train_index], Y_clib.loc[test_index]
                ### z-score env variables
                normaler.fit(X_train[env])
                X_train[env], X_test[env] = normaler.transform(X_train[env]), normaler.transform(X_test[env])
                ###
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    random_state=999,
                    n_jobs=12
                )
        ## ---------------------------------#####--------------------------------------
                model.fit(X_train, Y_train)
                # model = load(f'GBTs/gbt{i}_{randstate[i]}.joblib')  ### 改
                pred_train = model.predict(X_train)
                pred_test = model.predict(X_test)
                ##
                pred_clib = model.predict(
                    np.hstack(
                        (normaler.transform(X_clib[env]),X_clib[spa].to_numpy())
                    )
                )
                pred_validate = model.predict(
                    np.hstack(
                        (normaler.transform(X_validate[env]),X_validate[spa].to_numpy())
                    )
                )
                ##
                cv_results.iloc[i, 0], cv_results.iloc[i, 1], cv_results.iloc[i, 2] = r2_score(Y_train,pred_train), \
                                                                                   RMSE(Y_train,pred_train), \
                                                                                   mean_squared_error(Y_train,pred_train)
                cv_results.iloc[i, 3], cv_results.iloc[i, 4], cv_results.iloc[i, 5] = r2_score(Y_test,pred_test), \
                                                                                   RMSE(Y_test,pred_test), \
                                                                                   mean_squared_error(Y_test,pred_test)
                cv_results.iloc[i, 6], cv_results.iloc[i, 7], cv_results.iloc[i, 8] = r2_score(Y_clib,pred_clib), \
                                                                                   RMSE(Y_clib,pred_clib), \
                                                                                   mean_squared_error(Y_clib,pred_clib)
                cv_results.iloc[i, 9], cv_results.iloc[i, 10], cv_results.iloc[i, 11] = r2_score(Y_validate,pred_validate), \
                                                                                   RMSE(Y_validate,pred_validate), \
                                                                                   mean_squared_error(Y_validate,pred_validate)
                cv_results.iloc[i, 12], cv_results.iloc[i, 13], cv_results.iloc[i, 14] = n_estimators,max_features,j
                i+=1
            cv_results=cv_results.dropna().reset_index(drop=True)
            ## recording average outcomes ##
            optims = pd.DataFrame(cv_results.mean()).transpose()
            optims['algae']=j
            optims['seed']=M
            ## ----------------------- ##
            # optims = cv_results
            optim_keeper=pd.concat((optim_keeper,optims)).dropna()
            counts+=1
            if counts%5==0: print(counts,'in',j)
            optim_keeper.to_excel('RCB/optim_keeper_1994_RCB_2month.xlsx', index=False)  ### 改
## ---------------------------------#####--------------------------------------
# %%   model re-fitting
# optim_keeper = pd.read_excel('RCB/optim_keeper_1994_RCB_Sx.xlsx')
model_results = pd.DataFrame(index=range(0, 9999),
                             columns=['clib_r2', 'clib_rmse', 'clib_nrmse',
                                      'pred_r2', 'pred_rmse', 'pred_nrmse',
                                      'n_estimators', 'max_features', 'algae','seed'
                                      ])
i=0
for M in trange(0,1): #M=4,5,24, 40,296,231
    # M=126
    level_df_all= optim_keeper.loc[lambda x: x['seed'] ==40].reset_index(drop=True)
    base_clib, base_validate= train_test_split(base, test_size=0.2, random_state=40)
    base_clib=base_clib.reset_index(drop=True).fillna(0)
    base_validate=base_validate.reset_index(drop=True).fillna(0)

    normaler=StandardScaler()

    results_clib = pd.DataFrame()
    results_validate = pd.DataFrame()
    for j in algae:
        print(j)
        ### all
        env=np.concatenate([wq,metro_hydro])
        all_feature=np.concatenate([env,spa])
        X_clib=base_clib[all_feature]
        X_validate=base_validate[all_feature]
        ### selected
        # imp_x=imp.loc[lambda x: x['algae']==j].reset_index(drop=True)
        # env=np.concatenate((wq,imp_x['variable']))
        # important_feature=np.concatenate((env,spa))
        # X_clib=base_clib[important_feature]
        # X_validate=base_validate[important_feature]
        ###
        Y_clib = base_clib[algae][j]
        Y_validate = base_validate[algae][j]
        ##
        level_df= level_df_all.loc[lambda x: x['algae'] ==j].reset_index(drop=True)
        optim_record = level_df.sort_values('pred_rmse',ascending=True).reset_index(drop=True).iloc[0:1]
        ##
        n_estimators,max_features=round(optim_record['n_estimators'].item()),optim_record['max_features'].item()
        ##
        normaler.fit(X_clib[env])
        X_clib[env], X_validate[env] = normaler.transform(X_clib[env]), normaler.transform(X_validate[env])
        ##
        model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_features=max_features,
                        random_state=999,
                        n_jobs=12
                    ) #21 126 76
        model.fit(X_clib, Y_clib)
        pred_clib = model.predict(X_clib)
        pred_validate = model.predict(X_validate)
        ##
        results_clib = pd.concat((results_clib, pd.Series(pred_clib)), axis=1)
        results_validate = pd.concat((results_validate, pd.Series(pred_validate)), axis=1)

        ##stas_calculation
        model_results.iloc[i, 0], model_results.iloc[i, 1], model_results.iloc[i, 2] =r2_score(Y_clib,pred_clib), \
                                                                                       RMSE(Y_clib,pred_clib), \
                                                                                       NRMSE(Y_clib,pred_clib)
        model_results.iloc[i, 3], model_results.iloc[i, 4], model_results.iloc[i, 5] = r2_score(Y_validate,pred_validate), \
                                                                                       RMSE(Y_validate,pred_validate), \
                                                                                       NRMSE(Y_validate,pred_validate)
        model_results.iloc[i, 6], model_results.iloc[i, 7], model_results.iloc[i, 8], model_results.iloc[i, 9] = n_estimators, max_features, j, M
        i+=1
    ## models keeping  ### 改
    # model_list.append(dump(model, f'models/RF_models/AI+2.0/rf{i}_{randstate[i]}.joblib'))
model_results=model_results.round(decimals=3).dropna()
# %% training results recording
results_clib.columns=algae
results_clib=pd.concat((base_clib[algae],results_clib))
results_clib['type']=np.hstack((np.repeat('obs', len(Y_clib)),np.repeat('pred', len(Y_clib))))
results_clib['data']=np.repeat('clib', len(results_clib))
#
results_validate.columns=algae
results_validate=pd.concat((base_validate[algae],results_validate))
results_validate['type']=np.hstack((np.repeat('obs', len(Y_validate)),np.repeat('pred', len(Y_validate))))
results_validate['data']=np.repeat('validate', len(results_validate))
#
results=pd.concat((results_clib,results_validate))
# %% training statistics recording
model_results.to_excel('RCB/model_results_1994_2month.xlsx', index=False)
results.to_excel('RCB/results_1994_S.xlsx', index=False)
base_clib.to_excel('RCB/base_clib_1994_S.xlsx', index=False,sheet_name='base_clib')
base_validate.to_excel('RCB/base_validate_1994_S.xlsx', index=False,sheet_name='base_validate')

#%% cv_results keeper
# algae=np.array(['Cyanophyceae','Bacillariophyceae','Chrysophyceae','Cryptophyceae','phyto'])
# optim_keeper = pd.read_excel('optim_keeper_2000_S_all.xlsx')
cv_results = pd.DataFrame(index=range(0, 999),
                                 columns=['train_r2', 'train_rmse', 'train_nrmse',
                                          'test_r2', 'test_rmse', 'test_nrmse',
                                          'clib_r2', 'clib_rmse', 'clib_nrmse',
                                          'pred_r2', 'pred_rmse', 'pred_nrmse',
                                          'n_estimators','max_features','algae'
                                          ])
for M in trange(0,1): #M=4,5,24, 40,296,231
    # M=126
    level_df_all= optim_keeper.loc[lambda x: x['seed'] ==40].reset_index(drop=True)
    base_clib, base_validate= train_test_split(base, test_size=0.2, random_state=40)
    base_clib=base_clib.reset_index(drop=True).fillna(0)
    base_validate=base_validate.reset_index(drop=True).fillna(0)

    normaler=StandardScaler()
    counts=0
    i=0
    for j in algae:
        print(j)
        ### all
        # env=np.concatenate([wq,metro_hydro])
        # all_feature=np.concatenate([env,spa])
        # X_clib=base_clib[all_feature]
        # X_validate=base_validate[all_feature]
        ### selected
        imp_x=imp.loc[lambda x: x['algae']==j].reset_index(drop=True)
        env=np.concatenate((wq,imp_x['variable']))
        important_feature=np.concatenate((env,spa))
        X_clib=base_clib[important_feature]
        X_validate=base_validate[important_feature]
        ###
        Y_clib = base_clib[algae][j]
        Y_validate = base_validate[algae][j]
        ###
        level_df= level_df_all.loc[lambda x: x['algae'] ==j].reset_index(drop=True)
        optim_record = level_df.sort_values('pred_r2',ascending=False).reset_index(drop=True).iloc[0:1]
        ##
        n_estimators,max_features=round(optim_record['n_estimators'].item()),optim_record['max_features'].item()
        cv = KFold(n_splits=10,shuffle=True,random_state=999)

        for train_index, test_index in cv.split(X_clib):
            X_train, X_test = X_clib.loc[train_index], X_clib.loc[test_index]
            Y_train, Y_test = Y_clib.loc[train_index], Y_clib.loc[test_index]
            ### z-score env variables
            normaler.fit(X_train[env])
            X_train[env], X_test[env] = normaler.transform(X_train[env]), normaler.transform(X_test[env])
            ###
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=M,
                n_jobs=12
            )
    ## ---------------------------------#####--------------------------------------
            model.fit(X_train, Y_train)
            # model = load(f'GBTs/gbt{i}_{randstate[i]}.joblib')  ### 改
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            ##
            pred_clib = model.predict(
                np.hstack(
                    (normaler.transform(X_clib[env]),X_clib[spa].to_numpy())
                )
            )
            pred_validate = model.predict(
                np.hstack(
                    (normaler.transform(X_validate[env]),X_validate[spa].to_numpy())
                )
            )
            ##
            cv_results.iloc[i, 0], cv_results.iloc[i, 1], cv_results.iloc[i, 2] = r2_score(Y_train,pred_train), \
                                                                               RMSE(Y_train,pred_train), \
                                                                               mean_squared_error(Y_train,pred_train)
            cv_results.iloc[i, 3], cv_results.iloc[i, 4], cv_results.iloc[i, 5] = r2_score(Y_test,pred_test), \
                                                                               RMSE(Y_test,pred_test), \
                                                                               mean_squared_error(Y_test,pred_test)
            cv_results.iloc[i, 6], cv_results.iloc[i, 7], cv_results.iloc[i, 8] = r2_score(Y_clib,pred_clib), \
                                                                               RMSE(Y_clib,pred_clib), \
                                                                               mean_squared_error(Y_clib,pred_clib)
            cv_results.iloc[i, 9], cv_results.iloc[i, 10], cv_results.iloc[i, 11] = r2_score(Y_validate,pred_validate), \
                                                                               RMSE(Y_validate,pred_validate), \
                                                                               mean_squared_error(Y_validate,pred_validate)
            cv_results.iloc[i, 12], cv_results.iloc[i, 13], cv_results.iloc[i, 14] = n_estimators,max_features,j
            i+=1
        counts+=1
        if counts%5==0: print(counts,'in',j)

#%%
cv_results.dropna().to_excel('RCB/cv_results_1994_Sxx.xlsx', index=False)

#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.figure(1, figsize=(12, 5))
plt.cla()
red_patch = mpatches.Patch(color='red', label='predict')
blue_patch = mpatches.Patch(color='blue', label='true')
plt.legend(handles=[red_patch,blue_patch])
# plt.plot(clib_y,'b')
# plt.plot(clib_pred.mean(axis=1), 'r')
# plt.text(0,0, "r2: %1.5f" % (r2_score(clib_y,clib_pred.mean(axis=1))),
#          fontdict={'size': 15, 'color':  'red'})
plt.plot(pred_y,'b')
plt.plot(pred_pred, 'r')
plt.text(0,0, "r2: %1.5f" % (r2_score(pred_y,pred_pred)),
         fontdict={'size': 15, 'color':  'red'})

plt.title("train")
plt.show()