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

    
housing_xgb = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
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
#mesauring compute time
import time
start = time.time()
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
 
elapsed_time_fl = (time.time() - start)  
sns.scatterplot(data=rmse_df,x='Run',y='Rmse',color='blue')
sns.set_style("white")
plt.ylim(.11, .13)

#3rd pass
start2 = time.time()
rmse_df2 = pd.DataFrame()
for i in range(1,100):
    
    housing_xgb.fit(X_train[import_cols['cols'][:i]], y_train,verbose=True)
    
    pred = housing_xgb.predict(X_test[import_cols['cols'][:i]])                   
    rmse = np.sqrt(MSE(np.log(y_test+1), np.log(pred+1))) 
    #print("RMSE : % f" %(rmse)) 
       
    rss = sum((y_test -pred)**2)
    
    tss = sum((y_test - np.mean(y_test))**2)
    
    rsq  =  1 - (rss/tss)
    #print(rsq)
    rmse_df2 = rmse_df2.append(pd.DataFrame({'Run': len(import_cols['cols'][:i]),'Rmse': rmse, 'rsq': rsq},index=[0]),ignore_index=True)

elapsed_time_fl2 = (time.time() - start)      
sns.scatterplot(data=rmse_df2,x='Run',y='Rmse',color='blue')
sns.set_style("white")
plt.ylim(.11, .13)





# subset of top 70 variables
import_cols_70subset =import_cols[0:70]

 