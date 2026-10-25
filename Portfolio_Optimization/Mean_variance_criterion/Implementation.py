import numpy as np
import pandas as pd
from scipy.optimize import minimize
from Mean_variance_criterion import MV_criterion
from Import_data import data
from tabulate import tabulate


split = int( 0.7 * len (data) ) 
train_set = data.iloc[:split, :]
test_set = data.iloc[split:, :]
# Find the number of assets
n = data.shape[ 1 ] 
# Initialization weight value
x0 = np.ones(n ) 
# Optimization constraints problem
cons = ({ 'type' : 'eq' , 'fun' : lambda x: sum ( abs (x)) - 1 } ) 
# Set the bounds
Bounds = [( 0 , 1 ) for i in range ( 0 , n) ] 
# Optimization problem solving
res_MV = minimize(MV_criterion, x0, method= "SLSQP" ,
                  args=(train_set), bounds=Bounds,
                  constraints=cons, options={ 'disp' : True } ) 
# Result
X_MV = res_MV.x

# Print the optimized weights in table format
print(tabulate(zip(data.columns, X_MV), headers=["Asset", "Optimized Weight"], tablefmt="fancy_grid", floatfmt=".4f"))
