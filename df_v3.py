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

    
housing_xgb = xgb.XGBRegressor(objective='reg:linear', missing=False, 
                              seed=42,subsample=0.7)


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

# barplot of most important parms
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="cols", y="feat_import", data=import_cols[:30])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.title('Variables By Importance')
# Set x-axis label
plt.xlabel('Variable Names')
# Set y-axis label
plt.ylabel('Importance')
#2nd pass
#-----------------------------------------------------------------
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
plt.ylim(.134, .15)



#----------------------------------------------------------------------------------------
#2nd Pass but different model settings, 3500 trees

housing_xgb = xgb.XGBRegressor(objective='reg:linear',max_depth=3,missing=False, seed=42,
                               learning_rate = 0.01,n_estimators=3500, 
                               reg_alpha=0.00006,subsample=0.7)

rmse_3500_df = pd.DataFrame()

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
    rmse_3500_df = rmse_3500_df.append(pd.DataFrame({'Run': len(import_cols['cols'][:i]),'Rmse': rmse, 'rsq': rsq},index=[0]),ignore_index=True)

sns.scatterplot(data=rmse_3500_df,x='Run',y='Rmse',color='blue')
sns.set_style("white")
plt.ylim(.134, .15)

concatenated = pd.concat([rmse_df.assign(dataset='100_Trees'), rmse_3500_df.assign(dataset='3500_Trees')])

sns.scatterplot(x='Run',y='Rmse', data=concatenated,
                hue='dataset')
sns.set_style("white")

plt.title('Prediction Accuracy As A Function Of Features')
# Set x-axis label
plt.xlabel('# of Variables Used')
# Set y-axis label
plt.ylabel('Log Rmse')
plt.show()

#----------------------------------------------------------------------



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
plt.ylim(.134, .14)


# subset of top 70 variables
import_cols_70subset =import_cols[0:70]

 param_grid = { 
      'max_depth': [3, 4, 5],
      'learning_rate': [0.5, 0.2, 0.1],
      'gamma': [0, 0.25, 1.0],
      'reg_lambda': [0, 1.0, 10.0]
      }
# pass 3 
 param_grid = { 
      'max_depth': [3,4,6],
      'learning_rate': [0.0001,0.001,0.01,0.1],
      'subsample': [0.5,0.7,0.9],
      'n_estimators': [100,1000,3500,5000],
      'reg_alpha':[0.0001,0.00005,0.00001,0.000001]
      }

optimal_params = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:linear',seed=42),
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
    scoring='neg_mean_squared_log_error', ## see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=True, # NOTE: If you want to see what Grid Search is doing, set verbose=2
    n_jobs = 10,
    cv = 5
)



#optimal_params.fit(X_train[import_cols_70subset], y_train,verbose=True)
import time
startGS = time.time()

optimal_params.fit(X_train[import_cols[0:100]], y_train,verbose=True)

elapsed_time_GS = (time.time() - startGS)  

plt.hist(pred,bins=30)
plt.hist(y_test,bins=30)     
#     early_stopping_rounds=10,
#                eval_metric='acc',
#                eval_set=[(X_test, y_test)]
            
bst = housing_xgb.get_booster()     

#df.to_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/df_dummied.csv')


pred = optimal_params.predict(X_test)                   
rmse = np.sqrt(MSE(y_test, pred)) 
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



# grid search manually
from sklearn.model_selection import ParameterGrid


param_grid = { 
      'max_depth': [3,4,6],
      'learning_rate': [0.0001,0.001,0.01,0.1],
      'subsample': [0.5,0.7,0.9],
      'n_estimators': [100,1000,3500,5000],
      'reg_alpha':[0.0001,0.00005,0.00001,0.000001]
      }

# learning rate Study
param_grid = { 
      'max_depth': [3],
      'learning_rate': [0.0001],
      'subsample': [0.5],
      'n_estimators': list(range(0,5000,10)),
      'reg_alpha':[0.0001]
      }
pg=pd.DataFrame(ParameterGrid(param_grid))


import_cols=df_import[df_import['feat_import'] > 0.000]
import_cols=import_cols.sort_values(by='feat_import',ascending=False) # sort most important to least
rmse_df = pd.DataFrame()
#2nd pass

for i in range(0,len(pg)):
    
    housing_xgb = xgb.XGBRegressor(objective='reg:linear',max_depth=pg['max_depth'][i],missing=False, seed=42,
                               learning_rate = pg['learning_rate'][i],n_estimators=pg['n_estimators'][i], 
                               reg_alpha=pg['reg_alpha'][i],subsample=pg['subsample'][i])
    
    housing_xgb.fit(X_train[import_cols['cols'][:100]], y_train,verbose=True)
    #housing_xgb.fit(X_train[import_cols['cols']], y_train,verbose=True)
    
    pred = housing_xgb.predict(X_test[import_cols['cols'][:100]])
    #pred = housing_xgb.predict(X_test[import_cols['cols']])
    #pred = housing_xgb.predict(X_test)                     
    rmse = np.sqrt(MSE(np.log(y_test+1), np.log(pred+1))) 
    #print("RMSE : % f" %(rmse)) 
       
    rss = sum((y_test -pred)**2)
    
    tss = sum((y_test - np.mean(y_test))**2)
    
    rsq  =  1 - (rss/tss)
    #print(rsq)
    rmse_df = rmse_df.append(pd.DataFrame({'Run': i,'Rmse': rmse, 'rsq': rsq},index=[0]),ignore_index=True)
    print(i)

# matches test rmse values with xgboost parms
pg = pd.concat([pg, rmse_df], axis=1, join="inner")

pg.to_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/housing_proj/manual_GridSearch_test.csv',
              index_label=False)


pg_original= pd.read_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/housing_proj/manual_GridSearch2.csv')
pg=pg.sort_values(by='Rmse_100')
pg.to_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/housing_proj/manual_GridSearch2.csv',
              index_label=False)