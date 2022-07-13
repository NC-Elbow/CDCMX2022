#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:25:29 2021

@author: clark
"""

"""
This is a simple script to go through the time series forecasting test
given by PlusPower
"""

path = '/home/clark/Computing/python_projects/PlusPower_coding_test/'

import pandas as pd
import numpy as np


training = pd.read_csv(path + 'training.csv')

"""
Looks like we have 2208 blank rows
"""

print(training.isna().sum())

training = training.iloc[:-2208,:] #this removes the blanks space at the end ot the csv


def sepearate_date_into_features(df):
    date_col = df['Date']
    month = []
    day = []
    year = []
    for item in date_col:
        MDY = item.split('/') #due to the formatting of the training data set
        month.append(int(MDY[0]))
        day.append(int(MDY[1]))
        year.append(int(MDY[2])) #cleaner to insert than to assign columns in python 3.8
    df.insert(1, 'Date month', month)
    df.insert(2, 'Date day', day)
    df.insert(3, 'Date year', year)
    return df


"""
The temperature has 103 remaining nulls.  We'll do a simple linear 
interpolation to fill those points
"""
def get_numerical_df(df):
    numerical_columns = []
    for c in df.columns:
        if df[c].dtypes == int or df[c].dtypes == float:
            numerical_columns.append(c)
    numerical_df = df[numerical_columns]
    return numerical_df

def get_categorical_df(df):
    categorical_columns = []
    for c in df.columns:
        if df[c].dtypes == object:
            categorical_columns.append(c)
    categorical_df = df[categorical_columns]
    return categorical_df
    

def fill_data_midpoint(df):
    df_copy = get_numerical_df(df) #to avoid memory pointer issues
    df1 = df_copy.fillna(method = 'ffill')
    df1 = df1.fillna(method = 'bfill') #in case the first point is null
    df2 = df_copy.fillna(method = 'bfill')
    df2 = df2.fillna(method = 'ffill') # in case the last point is null
    df_copy = (df1 + df2)/2
    return df_copy

training = sepearate_date_into_features(training)
training.drop('Date', axis = 1, inplace = True)

training = fill_data_midpoint(training)

print(training.isna().sum())
"""
no nulls left after basic data cleaning
"""

#%%

"""
Let's get a little bit of feature engineering
"""

def compute_dart(df):
    dart = df['Day-Ahead LMP'] - df['Real-Time LMP']
    dart_direction = np.sign(dart.values)
    return dart, dart_direction

dart, dart_direction = compute_dart(training)
"""
It will be easier to repeat this on the test set
"""

training.insert(5,'dart', dart.values)
training.insert(6,'direction', dart_direction)

def convert_temperatures_F_to_C(temp):
    temp_f = temp
    temp_c = 5*(temp_f - 32)/9
    return temp_c

temp_c_boston = convert_temperatures_F_to_C(training['Temperature - Boston (deg. F)'])

training.insert(training.shape[1], 'Temperature - Boston (deg. C)', temp_c_boston)

#%%

target = 'Real-Time LMP'


import lightgbm as lgb

def split_train_test_by_random(df, training_proportion = 0.8):
    training_idx = np.random.choice(df.index, size = int(np.floor(training_proportion*df.shape[0])), replace = False)
    train_df = df.loc[training_idx]
    test_df = df.drop(training_idx)
    return train_df, test_df
    
def get_XY(df, target_name):
    train, test = split_train_test_by_random(df)
    X = train.drop(target_name, axis=1)
    Y = train[target_name]
    x = test.drop(target_name,axis=1)
    y = test[target_name]
    return X,x,Y,y

no_leak_training = training.drop(['dart','direction'],axis=1)

X,x,Y,y = get_XY(no_leak_training, target)
m = lgb.LGBMRegressor(n_leaves = 511, n_estimators= 501)
m.fit(X,Y)
y_pred = m.predict(x)
er = y_pred - y.values


import matplotlib.pyplot as plt

plt.plot(er)
plt.show()

plt.hist(er, bins = 100)
plt.show()





