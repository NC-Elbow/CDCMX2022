#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:21:41 2021

@author: clark
"""

"""
This module is to make a set of predictions for the plus power 
training and test data sets.
We assume they have been cleaned and feature engineered to our liking


This modeul will take in the training and test data sets separately to
avoid data leakage.  We will also get a particular target.

If the target is binary we will use a classifier model
if the target is continuous we will use a set of regression tools.

The only binary target we will consider for now is the 'direction' of the 
dart (day ahead -  real time)
"""

import pandas as pd
import numpy as np
from numpy import array as arr
from numpy.random import randn, rand, randint, choice
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA

class predictor:
    def __init__(self, training_data, testing_data, target_name):
        self.train = training_data
        self.test = testing_data
        self.target = target_name
        if len(set(self.test[self.target].values)) == 2:
            self.classification = True
            self.regression = False
        else:
            self.classification = False
            self.regression = True

    def get_XY(self, df, target_name):
        X = df.drop(target_name, axis=1)
        Y = df[target_name]
        return X, Y
 


    def build_boosted_tree_model(self, params_list):
        p = params_list
        if self.classification:
            tree_model = lgb.LGBMClassifier(boosting_type = p[0], 
                                            learning_rate=p[1],
                                            max_depth = p[2],
                                            num_leaves = p[3],
                                            n_estimators = p[4],
                                            reg_alpha = p[5],
                                            reg_lambda = p[6])
        else:
            tree_model = lgb.LGBMRegressor(boosting_type = p[0], 
                                           learning_rate=p[1],
                                           max_depth = p[2],
                                           num_leaves = p[3],
                                           n_estimators = p[4],
                                           reg_alpha = p[5],
                                           reg_lambda = p[6])
        return tree_model
    
    def get_tree_model_parameters(self):
        boosting = ['gbdt','rf','dart','goss'] # p0
        b_type = choice(boosting) # p0
        #class_wt = rand() # p1
        learning_rt = rand()/5 # p1
        depth = randint(3,8) # p2
        n_leaves = randint(depth + 1, 2**depth) # p3
        n_estimators = 2*randint(50,500)+1 # p4
        reg_alpha = rand() # p5
        reg_lambda = 5*rand() # p6
        params = [b_type, learning_rt, depth, n_leaves, n_estimators,
                  reg_alpha, reg_lambda]
        return params
    
    def score_tree_model(self, parameters):
        X,Y = self.get_XY(self.train, self.target)
        x,y = self.get_XY(self.test, self.target)
        m = self.build_boosted_tree_model(parameters)
        m.fit(X,Y)
        if self.classification:
            y_prob = m.predict_proba(x)[:,1]
            ts = np.linspace(0,1,51)
            fs = []
            for t in ts:
                y_pred = 1*(y_prob >= t)
                fs.append(f1_score(y, y_pred))
            score = np.max(fs)
        else:
            y_pred = m.predict(x)
            score = 1/(1+np.sum(np.abs(y_pred - y)))
        #print(score)    
        return score    
             
    def breed_params(self, params1, params2):
        child_params = [choice([params1[k],params2[k]]) for k in range(len(params1))]
        if (child_params[3] < child_params[2] + 2) or (child_params[2] >= 2**child_params[3]):
            child_params[3] = randint(child_params[2]+1, 2**child_params[2])
        return child_params

    def create_initial_population(self, size_of_population = 32):
        population = []
        for k in range(size_of_population):
            temp_params = self.get_tree_model_parameters()
            population.append(temp_params)
        return population

    def create_new_population(self, old_population, size_of_population = 16):
        scores = []
        new_pop = old_population.copy()
        for k in range(2*size_of_population):
            pair = choice(np.arange(len(old_population)), 2, replace = False)
            parent1 = list(old_population[pair[0]])
            parent2 = list(old_population[pair[1]])
            temp_p1 = self.breed_params(parent1, parent2)
            temp_p2 = self.breed_params(parent1, parent2)
            temp_p3 = self.get_tree_model_parameters()
            new_pop.append(temp_p1)
            new_pop.append(temp_p2)
            new_pop.append(temp_p3)
        for p in new_pop:
            try:
                scores.append(self.score_tree_model(p))
            except:
                scores.append(0)
        next_gen_locs = np.argsort(scores)[-size_of_population:] #looking for the best n solution     
        best_population = []
        for loc in next_gen_locs:
            best_population.append(new_pop[loc]) #if we convert to an array, the numerical items get converted to strings due to the 'gbdt' parameter
        new_pop = list(new_pop)
        return best_population, scores
    
    def evolve_to_solution(self, size_of_generation = 8, number_of_generations = 4):
        initial_population = self.create_initial_population(size_of_population = 8)
        new_population, scores = self.create_new_population(initial_population, 
                                                            size_of_population = size_of_generation)
        for k in range(1, number_of_generations):
            new_population, scores1 = self.create_new_population(new_population, 
                                                                 size_of_population = size_of_generation)
            print("Generation {0} is evolved".format(k+1))
            scores += scores1
        return new_population, scores    
            
"""
#test code: uncomment to run
train0 = plusPowerPrep(train_csv)
train0.clean_df()
train1 = train0.df

test0 = plusPowerPrep(test_csv)
test0.clean_df()
test1 = test0.df

tester = predictor(train1, test1, target_name = 'log dart')
#pop0 = tester.create_initial_population(size_of_population = 8)
#pop1,scores0 = tester.create_new_population(pop0, size_of_population = 4)

fit_population = tester.evolve_to_solution(size_of_generation = 4, number_of_generations = 3)


model = tester.build_boosted_tree_model(fit_population[-1])
X,Y = tester.get_XY(tester.train, tester.target)
x,y = tester.get_XY(tester.test, tester.target) 
model.fit(X,Y)
y_pred = model.predict(x)

"""
       
        


    
        