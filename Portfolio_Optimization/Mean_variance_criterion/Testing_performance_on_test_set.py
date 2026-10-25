# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:38:30 2025

@author: hp
"""
#Compute the cumulative return of the portfolio (CM)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Implementation import test_set , X_MV

portfolio_return_MV = np.multiply(test_set,np.transpose(X_MV))
portfolio_return_MV = portfolio_return_MV.sum(axis=1)

# Plot the CM

plt.figure(figsize=(15,8))
plt.plot(np.cumsum(portfolio_return_MV)*100, color="#035593" , linewidth= 3)
plt.ylabel( "Cumulative return %" , size= 15)
plt.xticks(size= 5 ,rotation = 45)
plt.yticks(size= 15 ,fontweight= 'bold' )
plt.title( "Cumulative return of the mean variance potfolio",size=20)
plt.axhline( 0 , color= "r" ,linewidth= 3 )
plt.show( )
