#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:08:58 2021

@author: pim01001
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/housing_proj/manual_GridSearch2.csv')

test=df.loc[(df['learning_rate'] == 0.01) & (df['subsample'] == 0.5) 
       & (df['reg_alpha'] == 0.0001) & (df['max_depth'] == 3)]

sns.set_theme(style="whitegrid")    
ax = sns.barplot(x="Rmse_100", y="n_estimators", data=test)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

sub_samp=df.loc[(df['n_estimators'] == 3500) 
       & (df['reg_alpha'] == 0.0001) & (df['max_depth'] == 3)]

sub_samp.to_csv('/home/pim01001/Documents/Bootcamp/python/proj/housing/sub_sample.csv')


sns.set_theme(style="whitegrid")
ax = sns.barplot(y="Rmse_100", x='subsample', hue='learning_rate',data=sub_samp)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


ax=sns.scatterplot(x="Rmse_100", y='subsample', hue='learning_rate',data=sub_samp)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


g = sns.catplot(
    data=sub_samp, kind="bar",
    x='subsample', y="Rmse_100", hue='learning_rate',
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")