#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:05:18 2020

@author: pim01001
"""

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)


housing = pd.read_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/train.csv')

colnames=pd.DataFrame(list(zip(housing.columns,housing.dtypes)),columns=['labels','tpes'])
colnames.to_csv(r'/home/pim01001/Documents/Bootcamp/python/proj/housing/columns.csv', index = False, header=True)

df = housing.copy()

df.drop('Id',axis=1, inplace=True)

#gives type of column
print(df.dtypes)

# gives a heat map of missing data
cols = df.columns[:80] # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))

 

# gets a list of colums which have na vlaues
na_val =[]

for i in df.columns:
    na_val.append(sum(df[i].isnull()))

#list(zip([df.columns,df.dtypes,na_val]))

combdf = pd.DataFrame({'Variable': df.columns,'Type':df.dtypes,'#_naVal':na_val})
combdf.reset_index(drop=True)
print(combdf.sort_values(by='#_naVal')[50:80])

combdf.to_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/columns_v2.csv',
              index_label=False)


# After review, fill all empty spaces with zero
col_names=['LotFrontage','Alley','BsmtQual','BsmtCond','BsmtExposure',
   'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageYrBlt',
   'GarageFinish','GarageQual','GarageCond',
   'PoolQC','Fence','MiscFeature']
df[col_names] = df[col_names].fillna(0)



cols = df.columns[:80] # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))

#found empty spaces Need to come back to this
#df['MasVnrArea']=df['MasVnrArea'][pd.isnull(df['MasVnrArea'])].fillna(0)
#df['Exterior2nd']=df['Exterior2nd'][pd.isnull(df['Exterior2nd'])].fillna(0)

cols = df.columns[30:80] # first 30 columns
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))


df.to_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/df.csv',
              index_label=False)


# 
#penguins = sns.load_dataset("penguins")
#sns.histplot(penguins, x="flipper_length_mm")
sns.histplot(df[df['SalePrice']> 300000],x='SalePrice',bins=100)

# number of data points

print(housing['SalePrice'].describe())




