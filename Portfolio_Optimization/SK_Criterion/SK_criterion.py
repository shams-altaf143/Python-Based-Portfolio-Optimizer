# -*- coding: utf-8 -*
def SK_criterion(weights, data):
    """
   -------------------------------------------------------------------------
   | Output: optimization portfolio criterion                                  |
   -------------------------------------------------------------------------
   | Inputs: -weight (type ndarray numpy): Weight for
portfolio               |
   |         -data (type ndarray numpy): Returns of stocks                     |
   -------------------------------------------------------------------------
   """
    from scipy.stats import skew, kurtosis
    import numpy as np
   #Parameter s1
    Lambda = 3
    W=1
    Wbar=1+0.25/100
   
   #Compute portfolio return s2 
    portfolio_return = np.multiply(data, np.transpose(weights))
    portfolio_return = portfolio_return. sum (axis= 1 )
   
   # Compute mean, volatility, skew, kurtosis of the portfolio
    mean = np.mean(portfolio_return,axis=0)
    std = np.std(portfolio_return,axis=0)
    skewness = skew(portfolio_return,0)
    kurt = kurtosis(portfolio_return,0)
   
   # Compute the criterion
    criterion = Wbar ** ( 1 - Lambda) / ( 1 + Lambda) + Wbar ** (Lambda) \
                * W * mean - Lambda / 2 * Wbar ** ( -1 - Lambda) * W ** 2 *std ** 2 \
                + Lambda * (Lambda + 1 ) / ( 6 ) * Wbar ** ( -2 - Lambda) *W ** 3 * \
                skewness- Lambda * (Lambda + 1 ) * (Lambda + 2 ) / ( 24 ) * Wbar ** (-3 - Lambda)\
                *W ** 4 * kurt
   
    criterion = - criterion
    return criterion