# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 17:07:32 2025

@author: hp
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from SR_criterion import SR_criterion
from Import_data import data
from tabulate import tabulate


split = int( 0.7 * len (data) ) 
train_set = data.iloc[:split, :]
test_set = data.iloc[split:, :]

 # Find the number of assets
n = data.shape[ 1 ]
 
# Initialization weight value
x0 = np.ones(n)
 
# Optimization constraints problem
cons = ({ 'type' : 'eq' , 'fun' : lambda x: sum ( abs (x)) - 1 })
 
# Set the bounds
Bounds = [( 0 , 1 ) for i in range ( 0 , n)]
 # Optimization problem solving
res_SR = minimize(SR_criterion, x0, method= "SLSQP" ,
                  args=(train_set), bounds=Bounds,
                  constraints=cons, options={ 'disp' : True })
 # Result for computations
X_SR = res_SR.x

print(tabulate(zip(data.columns, X_SR), headers=["Asset", "Optimized Weight"], tablefmt="fancy_grid", floatfmt=".4f"))