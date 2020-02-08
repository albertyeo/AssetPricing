#State Prices

'''
Calculation of pricing kernel, risk-neutral probability distribution, and
initial price of portfolio based on binomial-tree pricing for stocks
'''
#------------------------------------------------------------------------------
import numpy as np
#------------------------------------------------------------------------------
def StatePrices(period, k, R_f, X_i, P_i, phi, Y):
    #Pascal Tree Generator
    def gen(n,r=[]):
        for x in range(n):
            l = len(r)
            r = [1 if i == 0 or i == l else r[i-1]+r[i] for i in range(l+1)]
            yield r
        
    P_f = np.array([1 / R_f])
    P = np.append(P_f, P_i, axis = 0)
    
    X_f = np.array([1]*k).reshape(k, -1)
    X = np.append(X_f, X_i, axis = 1)
    
    p = P.T @ np.linalg.inv(X) 
    phi_hat = p * R_f
    
    M = p / phi
        
    P1 = np.sum([(p[0]**(period-i) * p[1]**(i) * Y[i]) * 
         (list(gen(period+1))[-1])[i] for i in range(len(Y))])
    
    P2 = np.sum([(((phi[0]*M[0])**(period-i)) * ((phi[1]*M[1])**(i)) * Y[i]) * 
         (list(gen(period+1))[-1])[i] for i in range(len(Y))])
    
    P3 = np.sum([(phi_hat[0]**(period-i) * phi_hat[1]**(i) * Y[i]) * 
         (list(gen(period+1))[-1])[i] for i in range(len(Y))])/(R_f**period)
    
    print('Pricing Kernel: ', M)
    print('Risk-Neutral Probability Distribution: ', phi_hat)
    print('Initial Price of Portfolio: ', P1)
    
    if period == 1:
        N = np.linalg.inv(X)@Y
        print('Required Shares in Each Asset: ', N)
    
    print('\n')
        
    return P1, P2, P3
#------------------------------------------------------------------------------
#p = Number of Period
#k = Number of States = 2
#X_i = Array of Asset Payoffs [k x k-1] (2D-Array)
#P_i = Array of Intial Prices [k-1 x 1] (1D-Array)
#phi = Physical Probability Distribution [1 x k] (1D-Array)
#Y = Array of Desired Payoffs (1D-Array)
#P1, P2, P3 = StatePrices(period, k , R_f, X_i, P_i, phi, Y)
