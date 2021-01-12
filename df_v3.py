#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:21:46 2020

@author: pim01001
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb # XGBoost stuff
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer # for scoring during cross validation
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import confusion_matrix # creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.metrics import mean_squared_log_error

#housing = pd.read_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/train.csv')

df = pd.read_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/df.csv')

# gets list of catagorical columns
cat_cols=list(df.dtypes[df.dtypes =='object'].index.values)




# filter data

df=df[df['SalePrice'] < 520000]


Y = df['SalePrice'].copy()
plt.hist(df['SalePrice'],bins=50)

df.drop('SalePrice',axis=1, inplace=True)
#one hot encoding




df=pd.get_dummies(df, columns=cat_cols)



cols = df.columns # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))

X_train, X_test, y_train, y_test = train_test_split(df, Y, random_state=42) #stratify = Y)


#1 filter data
#plt.hist(y_train,bins=50)
#y_train=y_train[y_train < 520000]

    
housing_xgb = xgb.XGBRegressor(objective='reg:linear', missing=False, n_estimators=1000,
                               seed=42,subsample=0.7,reg_alpha=0.00006)

# housing_xgb = xgb.XGBRegressor(objective='reg:linear',max_depth=3,missing=False, seed=42,
#                                learning_rate = 0.01,n_estimators=3460, 
#                                reg_alpha=0.00006,subsample=0.7)
# first pass
housing_xgb.fit(X_train, y_train,verbose=True)


# first pass 0.63
pred = housing_xgb.predict(X_test)                   
rmse = np.sqrt(MSE(np.log(y_test+1), np.log(pred+1))) 
print("RMSE : % f" %(rmse)) 
   
rss = sum((y_test -pred)**2)

tss = sum((y_test - np.mean(y_test))**2)

rsq  =  1 - (rss/tss)

#makes a list of featurew with thier importance
df_import = pd.DataFrame({'cols': X_test.columns,
                          'feat_import':pd.Series(housing_xgb.feature_importances_)})
#df_import.sort_values(by=['feat_import']).to_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/feat_import.csv')
# returns column names with importance greater than 0, 303 columns to 172

#import_cols=df_import[df_import['feat_import'] > 0.000]['cols'].values
import_cols=df_import[df_import['feat_import'] > 0.000]
import_cols=import_cols.sort_values(by='feat_import',ascending=False) # sort most important to least
rmse_df = pd.DataFrame()
#2nd pass
for i in range(1,len(import_cols)):
    
    housing_xgb.fit(X_train[import_cols['cols'][:i]], y_train,verbose=True)
    
    pred = housing_xgb.predict(X_test[import_cols['cols'][:i]])
    #pred = housing_xgb.predict(X_test)                     
    rmse = np.sqrt(MSE(np.log(y_test+1), np.log(pred+1))) 
    #print("RMSE : % f" %(rmse)) 
       
    rss = sum((y_test -pred)**2)
    
    tss = sum((y_test - np.mean(y_test))**2)
    
    rsq  =  1 - (rss/tss)
    #print(rsq)
    rmse_df = rmse_df.append(pd.DataFrame({'Run': len(import_cols['cols'][:i]),'Rmse': rmse, 'rsq': rsq},index=[0]),ignore_index=True)
    
sns.scatterplot(data=rmse_df,x='Run',y='Rmse',color='blue')
sns.set_style("white")
plt.ylim(.134, .16)

#3rd pass
rmse_df2 = pd.DataFrame()
for i in range(1,70):
    
    housing_xgb.fit(X_train[import_cols['cols'][:i]], y_train,verbose=True)
    
    pred = housing_xgb.predict(X_test[import_cols['cols'][:i]])                   
    rmse = np.sqrt(MSE(np.log(y_test+1), np.log(pred+1))) 
    #print("RMSE : % f" %(rmse)) 
       
    rss = sum((y_test -pred)**2)
    
    tss = sum((y_test - np.mean(y_test))**2)
    
    rsq  =  1 - (rss/tss)
    #print(rsq)
    rmse_df2 = rmse_df2.append(pd.DataFrame({'Run': len(import_cols['cols'][:i]),'Rmse': rmse, 'rsq': rsq},index=[0]),ignore_index=True)
    
sns.scatterplot(data=rmse_df2,x='Run',y='Rmse',color='blue')
sns.set_style("white")
plt.ylim(.11, .14)


# subset of top 70 variables
import_cols_70subset =import_cols[0:70]

param_grid = { 
    'max_depth': [3],
    'learning_rate': [0.0001,0.001],
 #    'gamma': [0, 1.0,5.0]
    'subsample': [.5,0.7,.9],
    'reg_alpha': [0.0001,0, 1.0],
    'n_estimators': [100,1000,2000,3000,4000]
    }
# pass 3 


optimal_params = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:linear', 
                                seed=42,
                                base_score=0.5),
    param_grid=param_grid,
    scoring='neg_mean_squared_log_error', ## see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=True, # NOTE: If you want to see what Grid Search is doing, set verbose=2
    n_jobs = 10,
    cv = 5
)




optimal_params = GridSearchCV(
    estimator=xgb.housing_xgb = xgb.XGBRegressor(objective='reg:linear', seed=42,
                               reg_alpha=0.00006,subsample=0.7,
    param_grid=housing_xgb.get_params,
    #scoring='neg_mean_squared_log_error', ## see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=True, # NOTE: If you want to see what Grid Search is doing, set verbose=2
    n_jobs = 10,
    cv = 3
)



optimal_params.fit(X_train[import_cols_70subset], y_train,verbose=True)

plt.hist(pred,bins=30)
plt.hist(y_test,bins=30)     
#     early_stopping_rounds=10,
#                eval_metric='acc',
#                eval_set=[(X_test, y_test)]
            
bst = housing_xgb.get_booster()     

#df.to_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/df_dummied.csv')


pred = optimal_params.predict(X_test)                   
#rmse = np.sqrt(MSE(y_test, pred)) 
rmse2=mean_squared_log_error(y_test, pred)
rmse=np.sqrt(MSE(np.log(y_test+1), np.log(pred+1))) 
print("RMSE : % f" %(rmse)) 
   
rss = sum((y_test -pred)**2)

tss = sum((y_test - np.mean(y_test))**2)

rsq  =  1 - (rss/tss)
print(rsq)


housing_xgb2 = xgb.XGBRegressor(objective='reg:linear', gamma= 0,
learning_rate= 0.3,max_depth= 3,missing=True, seed=42)

housing_xgb2.fit(X_train[import_cols_70subset], y_train,verbose=True)
pred=housing_xgb2.predict(X_test) 
rmse = np.sqrt(MSE(y_test, pred)) 
print("RMSE : % f" %(rmse)) 

