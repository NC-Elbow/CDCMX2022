#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:24:51 2021

@author: clark
"""

"""
This is the data cleaning module for plus power.
We will be able to clean both the training and testing files
"""
import numpy as np
import pandas as pd

class plusPowerPrep:
    def __init__(self, df):
        self.df = df
        self.null_counts = self.df.isna().sum()
        if self.null_counts['Hour'] > 0:
            self.df = self.df.iloc[: -self.null_counts['Hour'] , :]
            """
            Apparently the train and test csvs were separated at sept 30, 2020
            and october 1, 2020, so the train csv has 2208 nulls which
            were not cleared.  Thus we take care of them here in the initializataion
            """
        
    def sepearate_date_into_features(self):
        if self.df.index.name == 'Date':
            date_col = self.df.index
        else:
            date_col = self.df['Date']
        month = []
        day = []
        year = []
        for item in date_col:
            MDY = item.split('/') #due to the formatting of the training data set
            month.append(int(MDY[0]))
            day.append(int(MDY[1]))
            year.append(int(MDY[2])) #cleaner to insert than to assign columns in python 3.8
        self.df.insert(1, 'Date month', month)
        self.df.insert(2, 'Date day', day)
        self.df.insert(3, 'Date year', year)
       
    def get_numerical_df(self):
        numerical_columns = []
        for c in self.df.columns:
            if self.df[c].dtypes == int or self.df[c].dtypes == float:
                numerical_columns.append(c)
        self.numerical_df = self.df[numerical_columns]
        

    def get_categorical_df(self):
        categorical_columns = []
        for c in self.df.columns:
            if self.df[c].dtypes == object:
                categorical_columns.append(c)
        self.categorical_df = self.df[categorical_columns]
        
    def fill_data_midpoint(self, df):
        self.get_numerical_df() #to avoid memory pointer issues
        df_copy = self.numerical_df.copy()
        df1 = df_copy.fillna(method = 'ffill')
        df1 = df1.fillna(method = 'bfill') #in case the first point is null
        df2 = df_copy.fillna(method = 'bfill')
        df2 = df2.fillna(method = 'ffill') # in case the last point is null
        df_copy = (df1 + df2)/2
        return df_copy    
    
    def convert_temperatures_F_to_C(self, temp_f):
        temp_c = 5*(temp_f - 32)/9
        return temp_c

    def convert_temperatures_F_to_R(self, temp_f):
        temp_r = temp_f + 459.67
        return temp_r

    def convert_temperatures_F_to_K(self, temp_f):
        temp_k = 5*(temp_f - 32)/9 + 273.15
        return temp_k

    def make_new_targets(self):
        if 'Real-Time LMP' in self.df.columns:
            RT_LMP = self.df['Real-Time LMP']
        else:
            RT_LMP = np.zeros(self.df.shape[0])
        DA_LMP = self.df['Day-Ahead LMP']
        DART = DA_LMP - RT_LMP
        direction = np.sign(DART.values)
        log_dart = direction*np.log(1 + np.abs(DART.values))
        z_dart = (DART.values - np.mean(DART.values))/(np.std(DART.values) + 1e-9) #we always add 1e-9 to avoid dividion by zero.
        self.df.insert(self.df.shape[1], 'dart', DART.values)
        self.df.insert(self.df.shape[1], 'direction', direction)
        self.df.insert(self.df.shape[1], 'log dart', log_dart)
        self.df.insert(self.df.shape[1], 'z dart', z_dart)
        
    def scale_by_mean(self, df):
        mu = df.mean()
        mdf = df/mu
        return mdf, mu

    def rescale_by_mean(self, mdf, mu):
         return mdf*mu

    def get_z_score(self, df):
        mu = df.mean()
        sig = df.std() + 1e-9
        zdf = (df - mu)/sig
        return zdf, mu, sig
    
    def restore_z_score(self, zdf, mu, sig):
        return mu + sig*zdf
    
    def scale_by_log(self, df):
        ldf = np.sign(df)*np.log(1 + np.abs(df))
        return ldf 
    
    def restore_log_scale(self, df):
        x = np.sign(df)*(np.exp(np.abs(df)) -  1)
        return x

    def difference_df(self, df, lag = 1):
        idx = df.index
        dfv = df.values
        diff_dfv = dfv[lag:,:] - dfv[:-lag, :]
        df_lag = pd.DataFrame(diff_dfv, columns = df.columns, index = idx[lag:])
        return df_lag

    def clean_df(self):
        self.sepearate_date_into_features()
        self.df.index = self.df['Date']
        self.df = self.df.drop('Date', axis = 1)
        self.fill_data_midpoint(self.df)
        temp_f = self.df['Temperature - Boston (deg. F)'].values
        temp_c = self.convert_temperatures_F_to_C(temp_f)
        self.df.insert(self.df.shape[1], 'Celsius', temp_c)
        temp_r = self.convert_temperatures_F_to_R(temp_f)
        self.df.insert(self.df.shape[1], 'Rankine', temp_r)
        temp_k = self.convert_temperatures_F_to_C(temp_f)
        self.df.insert(self.df.shape[1], 'Kelvin', temp_k)
        self.make_new_targets()