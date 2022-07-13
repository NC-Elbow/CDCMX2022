#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:10:50 2021

@author: clark
"""

"""
This will run and plot and give some analysis on the pluPower training sets
"""
import pandas as pd
import numpy as np
from numpy import array as arr 
from plusPowerDataPrep import plusPowerPrep
from plusPowerPredictor import predictor
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

train_csv = pd.read_csv('/home/clark/Computing/python_projects/PlusPower_coding_test/training.csv')
test_csv = pd.read_csv('/home/clark/Computing/python_projects/PlusPower_coding_test/test.csv')

train0 = plusPowerPrep(train_csv)
test0 = plusPowerPrep(test_csv)

train0.clean_df()
train1 = train0.df

test0.clean_df()
test1 = test0.df

additional_targets = ['dart','direction', 'log dart', 'z dart']

train2 = train1.drop(additional_targets, axis = 1)
test2 = test1.drop(additional_targets, axis = 1)

def log_transform(vec):
    y = np.sign(vec)*np.log(1+np.abs(vec))
    return y

def unwind_log_transform(vec):
    x = np.sign(vec)*(np.exp(np.abs(vec))-1)
    return x

train3 = log_transform(train2)
test3 = log_transform(test2)

ga_predictor = predictor(train3, test3, target_name = 'Real-Time LMP')
"""
A note here that the real time marginal price is log transformed.
To get the real prediction back, we'll have to unwind the predictions.
"""


"""
Now we will go through a very tiny genetic algorithm to fit 
a gradient boosted tree's model parameters for a log transformed
locational marginal price prediction
"""

fit_population, fitnesses = ga_predictor.evolve_to_solution(size_of_generation = 12, number_of_generations = 6)

#%%

plt.figure()
plt.plot(fitnesses, label = 'raw')
plt.plot(sorted(fitnesses), label='sorted fitnesses')
plt.xlabel('total set of fitness for predictions')
plt.show()


best_parameters = fit_population[-1]
print("The best parameters found on this run were {0}".format(best_parameters))

"""
To save ourselves time in the future, here is a set of parameters that we 
came up with:
    
best_parameters = ['goss', 0.052735767178120584, 3, 5, 161, 0.5212634866630119, 4.06976587223678]



"""

counter = 1
y_pred = arr([])
while len(y_pred) == 0:
    model = ga_predictor.build_boosted_tree_model(fit_population[-counter])
    X,Y = ga_predictor.get_XY(ga_predictor.train, ga_predictor.target)
    x,y = ga_predictor.get_XY(ga_predictor.test, ga_predictor.target) 
    try:
        model.fit(X,Y)
        y_pred = model.predict(x)
    except:
        y_pred = arr([])
        counter += 1

plt.figure()
plt.plot(y_pred, label = 'log predicitions')
plt.plot(y.values, label = 'log actual values')
plt.legend()
plt.show()

plt.figure()
plt.plot(y_pred - y.values, label = 'predictions - actuals')
plt.plot(np.abs(y_pred - y.values), label = 'absolute difference')
plt.legend()
plt.xlabel("Logarithmic predictions")
plt.show()

real_time_y_pred = unwind_log_transform(y_pred)
real_time_y = test1['Real-Time LMP'].values

plt.figure()
plt.plot(real_time_y_pred, label = 'log predicitions')
plt.plot(real_time_y, label = 'actual values')
plt.legend()
plt.show()

plt.figure()
plt.plot(real_time_y_pred - real_time_y, label = 'predictions - actuals')
plt.plot(np.abs(real_time_y_pred - real_time_y), label = 'absolute difference')
plt.xlabel("Unwound predictions")
plt.legend()
plt.show()

#%%
"""
To predict a particular period

We will have a test set with date day ahead lmp, etc in csv format

tomorrow_data = pd.read_csv('./tomorrow_data.csv')
test = tomorrow_data.clean_df()
data_to_predict = test.df
x,y = ga_predictor.get_XY(data_to_predict, target_name)
y_pred = model.predict(x)

"""


