# -*- coding: utf-8 -*
def SR_criterion(weight, data):
        """
   -------------------------------------------------------------------------
   | Output: Opposite Sharpe ratio to minimize it.                          |
   -------------------------------------------------------------------------
   | Inputs: -Weight (type ndarray numpy): Weight for
portfolio               |
   |         -data (type dataframe pandas): Returns of stocks                  |
   -------------------------------------------------------------------------
        """
        import numpy as np
   #Parameter
        rf=0 #  is the risk-free asset (we set 0 for the risk-free assets because it is currently around this value).
   # Compute portfolio returns
        portfolio_return = np.multiply(data, np.transpose(weight))
        portfolio_return = portfolio_return. sum (axis= 1 )

   # Compute mean, volatility of the portfolio
        mean = np.mean(portfolio_return, axis= 0 )
        std = np.std(portfolio_return, axis= 0 )

   # Compute the opposite of the Sharpe ratio
        Sharpe = (mean-rf) / std
        Sharpe = -Sharpe
        return Sharpe