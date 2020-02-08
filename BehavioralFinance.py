#Behavioral Finance

'''Monte Carlo simulation of utility function based on Barberis, Huang, and 
Santos (2001) economy to calculate the price-dividend ratio for market 
portfolio and expected market return. The results are plotted against the 
amount of emphasis that investor puts on utility from financial gain or loss, 
compared to the utility of consumption
'''
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
def BF(theta, gamma, lumbda, array_rf):
    #Consumption growth function
    g = np.array([np.exp(0.02 + 0.02 * np.random.normal()) 
                for i in range (10000)])
    
    #Utility from financial gain or loss
    def nu_R(R):
        return [i - array_rf if i >= array_rf else lumbda * (i - array_rf) 
                for i in R]
    
    #Error term
    def e_term(x, b_0):
        return theta * b_0 * np.mean(nu_R(x*g)) + theta * x - 1
    
    #Bisection search
    b_0 = np.linspace(0, 10, 101)
    x_values = []
    def x_term(x_neg, x_pos, b_0):
        x = 0.5 * (x_neg + x_pos)
        if np.abs(e_term(x, b_0)) < 10**(-4):
            x_values.append(x)
        elif e_term(x, b_0) < 0:
            x_term(x, x_pos, b_0)
        elif e_term(x, b_0) > 0:
            x_term(x_neg, x, b_0)
    
    for i in b_0:
        x_term(1, 1.1, i)
    
    #Price-Dividend ratio
    n = len(x_values)
    pd_ratio = [1 / (x_values[i] - 1) for i in range(n)]
    plt.plot(b_0, pd_ratio)
    plt.xlabel('b0')
    plt.ylabel('Price-Dividend Ratio')
    plt.show()
    
    #Expected market return
    market_return = np.array([np.mean(x_values[i] * g) for i in range(n)])
    equity_premium = market_return - array_rf
    
    plt.plot(b_0, equity_premium)
    plt.xlabel('b0')
    plt.ylabel('Equity Premium')
    plt.show()
#------------------------------------------------------------------------------
#BF(theta, gamma, lumbda, array_rf)
