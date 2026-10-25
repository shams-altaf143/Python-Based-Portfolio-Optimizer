import numpy as np

def MV_criterion ( weights , data):
    """
    -------------------------------------------------------------------------
    | Output: optimization portfolio criterion                                   |
    -------------------------------------------------------------------------
    | Inputs: -weight (type ndarray numpy): Weight for
 portfolio               |
    |         -data (type ndarray numpy): Returns of stocks                     |
    -------------------------------------------------------------------------
    """
 
    #Parameters 1 
    Lambda = 3
    W = 1
    Wbar = 1 + 0.25 / 100  #The risk-free wealth is 1 + the risk-free rate (0.25%)
 
    # Compute portfolio returns 
    portfolio_return = np.multiply(data, np.transpose(weights))  #Multiply the columns by their coefficient to keep a matrix with shape (n,m)
    portfolio_return = portfolio_return.sum(axis =1)
 
    # Compute mean and volatility of the portfolio
    mean = np.mean(portfolio_return, axis= 0 ) 
    std = np.std(portfolio_return, axis= 0 ) 
 
    # Compute the criterion
    criterion = (Wbar**(1-Lambda)/(1+Lambda)) + (Wbar**(-Lambda)*W*mean) - (Lambda/2)*(Wbar**(-1-Lambda))*(W**2)*(std**2)
 
    criterion = -criterion 
    return criterion

# Mean_variance_criterion.py

# import numpy as np

# def MV_criterion(weights, data, Lambda=3):
#     """
#     Mean‐Variance utility to *minimize* (negative of the criterion).
    
#     weights : ndarray, shape (n_assets,)
#     data    : DataFrame or ndarray, shape (n_samples, n_assets)
#     Lambda  : risk aversion parameter (higher → more risk‐averse)
#     """

#     # risk‐free wealth factor
#     W = 1
#     Wbar = 1 + 0.25 / 100  # 0.25% risk‐free rate

#     # portfolio returns (n_samples,)
#     # data can be DataFrame or numpy array
#     R = np.dot(data, weights)

#     mean = np.mean(R)
#     std  = np.std(R)

#     # utility criterion
#     utility = (Wbar**(1-Lambda)/(1+Lambda)
#                + Wbar**(-Lambda)*W*mean
#                - (Lambda/2)*(Wbar**(-1-Lambda))*(W**2)*(std**2))

#     # we minimize −utility to maximize utility
#     return -utility





# 1 Set the parameters of the model above.
#  2The risk-free wealth is 1 + the risk-free rate
#  (0.25%).
# 3 Multiply the columns by their coefficient to keep a
#  matrix with shape (n,m).
#  4 Sum all the columns to have the portfolio (shape=
#  (n,1)).
#  5 Compute the mean of the portfolio daily returns
#  (using axis=0 to do on the rows).
#  6 Compute the daily volatility of the portfolio (the
#  standard deviation).
#  7 Compute using the previous formula.
#  8 Return the opposite of the criterion to minimize it.

#  Maximizing a function is the same as minimizing the
#  opposite of this function. So, we will use the minimize
#  function in the following code because the
#  MV_criterion returns us  