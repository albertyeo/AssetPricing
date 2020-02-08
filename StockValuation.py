#Stock Valuation

'''
Calculation of performance metrics, namely:
Sharpe Ratio, Treynor Ratio, Information Ratio, Sortino Ratio, Jensen's Alpha,
and Fama-French 3-Factor's Alpha to evaluate stocks
'''
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#------------------------------------------------------------------------------
#Market Model
def MarketModel(data_portfolio, array_rf, data_market):
    #Market Model
    x = data_market - array_rf
    y = data_portfolio - array_rf
    
    c = len(y.columns)
    
    #Regression of excess return of portfolio on market
    model = [LinearRegression().fit(x, y.iloc[:,i]) for i in range(c)]
    r_sq = [model[i].score(x, y.iloc[:,i]) for i in range(c)]
    alpha = [model[i].intercept_ for i in range(c)]
    beta = [model[i].coef_[0] for i in range(c)]
    
    MM_table = pd.DataFrame({'Alpha': alpha, 'Beta': beta, 'R2': r_sq}, 
                              index = y.columns)
    
    return MM_table
#------------------------------------------------------------------------------
#Fama-French 3 Factor Model
def FF3Factor(data_portfolio, array_rf, data_market, data_smb, data_hml):
    #Fama-French 3-Factor Model
    data_factors = pd.DataFrame((data_market-array_rf).values, 
                                columns = ['Rm-Rf'],
                                index = data_market.index)
    data_factors['SMB'] = data_smb.values
    data_factors['HML'] = data_hml.values
    
    x = data_factors[['Rm-Rf','SMB', 'HML']].values.reshape(-1 ,3)
    y = data_portfolio - array_rf
    
    c = len(y.columns)
    
    #Regression of excess return of portfolio on the 3 factors
    model = [LinearRegression().fit(x, y.iloc[:,i]) for i in range(c)]
    r_sq = [model[i].score(x, y.iloc[:,i]) for i in range(c)]
    alpha = [model[i].intercept_ for i in range(c)]
    beta = [model[i].coef_[0] for i in range(c)]
    
    FF_table = pd.DataFrame({'Alpha': alpha, 'Beta': beta, 'R2': r_sq}, 
                              index = y.columns)
    
    return FF_table
#------------------------------------------------------------------------------
#Performance Metrics
def PerformanceMetrics(data_portfolio, array_rf, data_market, 
                       data_smb, data_hml, array_target):
    #Excess return with rf
    excess_rf = data_portfolio - array_rf
    var_rf = np.var(excess_rf)

    #Excess return with target
    excess_target = data_portfolio - array_target
    var_target = np.var(excess_target)
    semivar_target = np.mean(np.minimum(excess_target, 0)**2)

    #Market Model Regression
    MM_table = MarketModel(data_portfolio, array_rf, data_market)

    #FF 3 Factor Model Regression
    FF_table = FF3Factor(data_portfolio, array_rf, data_market, data_smb, data_hml)

    #Metrics
    sharpe = np.mean(excess_rf) / np.sqrt(var_rf)
    treynor = np.mean(excess_rf) / MM_table['Beta']
    information = np.mean(excess_target) / np.sqrt(var_target)
    sortino = np.mean(excess_target) / np.sqrt(semivar_target)
    jensen_alpha = MM_table['Alpha']
    threefactor_alpha = FF_table['Alpha']
    
    metrics_table = pd.DataFrame({' Sharpe Ratio': sharpe,
                                  ' Treynor Ratio': treynor,
                                  'Information Ratio': information,
                                  'Sortino Ratio': sortino,
                                  "Jensen's Alpha": jensen_alpha,
                                  "FF3Factor's Alpha": threefactor_alpha},
                                index = data_portfolio.columns)
    
    m = len(metrics_table.columns)
    bar_plot = [metrics_table.iloc[:,[i]].plot(kind='bar', 
                title=metrics_table.columns[i], 
                legend = None) for i in range(m)]
    plt.show()

    return metrics_table
#------------------------------------------------------------------------------
#data_portfolio = pd.read_csv('name.csv', index_col = 0, header = 0)
#array_rf = DataFrame.values.reshape(-1,1)
#data_market = DataFrame[['Rm-Rf']] + array_rf
#data_smb = DataFrame[['SMB']]
#data_hml = DataFrame[['HML']]
#array_target = DataFrame.values.reshape(-1,1)
#------------------------------------------------------------------------------
#FF_table = FF3Factor(data_portfolio, array_rf, data_market, data_smb, data_hml)
#------------------------------------------------------------------------------
#metrics_table = PerformanceMetrics(data_portfolio, array_rf, data_market, data_smb, data_hml, array_target)
#print('----------------Performance Metrics Table--------------')
#print(metrics_table.iloc[:,0:3])
#print(metrics_table.iloc[:,3:])
