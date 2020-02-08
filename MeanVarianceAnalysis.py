#Mean-Variance Analysis

'''
Calculation of mean returns, standard deviation of returns, and weight of
tangency portfolio based on returns data. Plot of mean against variance is
generated alongside the minimum-variance frontier for both portfolios without
riskless assets and portfolios with riskless assets. Calculation of optimal
asset allocation based on CARA (Constant absolute risk aversion) is also done
'''
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
#Minimum-Variance Frontier
def MVF(data_portfolio, array_rf, y0, yt, num):    
    r_vector = data_portfolio.apply(np.mean, axis = 0)
    cov_matrix = data_portfolio.cov()
    n = len(r_vector.index)
    e_vector = np.ones((n, 1), dtype=int)
    cov_inverse = np.linalg.inv(cov_matrix)
    
    #Alpha, Zeta, Theta
    alpha = (r_vector.T @ cov_inverse @ e_vector)[0]
    zeta = r_vector.T @ cov_inverse @ r_vector
    theta = (e_vector.T @ cov_inverse @ e_vector)[0]
    
    #Without Riskless Asset
    r_p1 = np.linspace(0, yt, num)
    sigma1 = np.sqrt((1 / theta) + 
            (theta / (zeta * theta - alpha ** 2 )) * 
            (r_p1 - alpha / theta) ** 2)
    
    mvf1 = pd.DataFrame(r_p1, columns = ['Mean Returns'])
    mvf1['Sigma'] = sigma1

    #With Riskless Asset(WRA)
    r_p2 = np.linspace(y0, yt, num)
    sigma2 = np.sqrt((r_p2 - array_rf) ** 2 / 
             (zeta - 2 * alpha * array_rf + theta * array_rf ** 2))
    
    mvf2 = pd.DataFrame(r_p2, columns = ['Mean Returns WRA'])
    mvf2['Sigma WRA'] = sigma2
    
    #Tangency Portfolio
    r_mv = alpha / theta
    r_tg = r_mv - ((zeta * theta - alpha ** 2) / (theta ** 2 * (array_rf - r_mv)))
    a = (zeta * cov_inverse @ e_vector - 
         alpha * cov_inverse @ r_vector) / (zeta * theta - alpha ** 2)
    b = (theta * cov_inverse @ r_vector - 
         alpha * cov_inverse @ e_vector) / (zeta * theta - alpha ** 2)
    weight_tg = a + b * r_tg
    rp_tg = r_tg - array_rf
    sigma_tg = -(zeta - 2*alpha*array_rf + theta*array_rf**2)**(1/2) / (theta*(array_rf - r_mv))
    sharpe_tg = rp_tg / sigma_tg
    
    #Summary Table
    var_vector = [cov_matrix.iloc[i,i] for i in range(n)]
    table = pd.DataFrame(r_vector, 
                         index = r_vector.index, 
                         columns = ['Mean Returns'])
    table['SD of Returns'] = np.sqrt(var_vector)
    table['Weight Tangency'] = weight_tg[0]
    
    print('Risk premium for tangency portfolio: ', rp_tg[0])
    print('Sharpe ratio for tangency portfolio: ', sharpe_tg[0])
    print('\n')
    
    return mvf1, mvf2, table
#------------------------------------------------------------------------------
#Plot of MVF
def MVFPlot(mvf1, mvf2):
    #Without Riskless Asset
    plt.figure()
    plt.ylabel('Mean Return (%)')
    plt.xlabel('Standard Deviation of Return (%)')
    plt.title('Minimum-Variance Frontier (Without Riskless Asset)')
    plt.plot(mvf1.loc[:, 'Sigma'], mvf1.loc[:,'Mean Returns'], 'b-', 
             label = 'Without Riskless Asset')
    
    #With Riskless Asset
    plt.figure()
    plt.ylabel('Mean Return (%)')
    plt.xlabel('Standard Deviation of Return (%)')
    plt.title('Minimum-Variance Frontier (With Riskless Asset)')
    plt.plot(mvf2.loc[:, 'Sigma WRA'], mvf2.loc[:,'Mean Returns WRA'], 'r-', 
             label = 'With Riskless Asset')
    
    #Combined
    plt.figure()
    plt.ylabel('Mean Return (%)')
    plt.xlabel('Standard Deviation of Return (%)')
    plt.title('Minimum-Variance Frontier')
    plt.plot(mvf1.loc[:, 'Sigma'], mvf1.loc[:,'Mean Returns'], 'b-', 
             label = 'Without Riskless Asset')
    plt.plot(mvf2.loc[:, 'Sigma WRA'], mvf2.loc[:,'Mean Returns WRA'], 'r-', 
             label = 'With Riskless Asset')
    plt.legend(loc='upper left')
    plt.show()
#------------------------------------------------------------------------------
#CARA Utility
def CARA(data_portfolio, array_rf, b, W0):
    b_r = b * W0
    
    r_vector = data_portfolio.apply(np.mean, axis = 0)
    cov_matrix = data_portfolio.cov()
    n = len(r_vector.index)
    e_vector = np.ones((n, 1), dtype=int)
    cov_inverse = np.linalg.inv(cov_matrix)
    
    alpha = (r_vector.T @ cov_inverse @ e_vector)[0]
    theta = (e_vector.T @ cov_inverse @ e_vector)[0]
    
    weight_CARA = cov_inverse @ (r_vector - array_rf @ e_vector) / b_r
    invest_risky = (alpha - theta * array_rf) / b
    r_p = array_rf + weight_CARA.T @ (r_vector - array_rf @ e_vector)
    utility = -np.exp(-b_r * r_p)
    
    print('Optimal portfolio weights: ', weight_CARA)
    print('Absolute amount of wealth invested in risky assets: ', invest_risky)
    print('Portfolio return: ', r_p)
    print('Utility of final wealth: ', utility)
#------------------------------------------------------------------------------
#data_portfolio = df of returns
#array_rf = (-1, 1) array of risk-free returns or scalar
#y0 =
#yt =
#num =
#------------------------------------------------------------------------------  
#mvf1, mvf2, table = MVF(data_portfolio, array_rf, y0, yt, num)
#print(table)
#MVFPlot(mvf1, mvf2)
#------------------------------------------------------------------------------
#b = Coefficient of absolute risk aversion
#W0 = Initial wealth
#CARA(data_portfolio, array_rf, b, W_0)
