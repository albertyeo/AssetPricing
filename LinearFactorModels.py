#Linear Factor Models

'''
Calculation of alpha, beta, and coefficient of determination and plot of SML
based on returns data. The models used are Market Model, CAPM, Fama-French 
3-Factor Model, and Carhart 4-Factor Model
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
#CAPM
def CAPM(data_portfolio, array_rf, data_market, MM_table, x0, xt, num):
    #CAPM
    MM_table = MarketModel(data_portfolio, array_rf, data_market)
    r_portfolio = data_portfolio.apply(np.mean, axis = 0)
    r_market = data_market.apply(np.mean, axis = 0)
    
    CAPM_table = pd.DataFrame(r_portfolio, index = data_portfolio.columns, 
                               columns = ['Mean Return'])
    CAPM_table['Beta'] = MM_table['Beta']
    CAPM_table = CAPM_table.append(pd.DataFrame({'Mean Return': r_market, 
                                                   'Beta': 1}, 
                                                    index = ['Market']))
    
    #Regression of mean return of portfolio&market on beta
    x2 = CAPM_table['Beta'].values.reshape((-1, 1))
    y2 = CAPM_table['Mean Return'].values
    model2 = LinearRegression().fit(x2, y2)
    r_sq2 = model2.score(x2, y2)
    alpha2 = model2.intercept_
    beta2 = model2.coef_[0]   
    
    #Plot of SML
    x_SML = np.linspace(x0, xt, num)
    y_SML = alpha2 + beta2*x_SML
    
    plt.figure()
    plt.xlabel('Beta')
    plt.ylabel('R (%)')
    plt.plot(x_SML, y_SML, 'r-', label ='SML')
    plt.legend(loc = 'upper left')
    plt.scatter(x2, y2, color='blue')
    plt.show()
    
    print('Alpha of CAPM: ', np.round(alpha2,4))
    print('Beta of CAPM: ', np.round(beta2,4))
    print('R2 of CAPM: ', np.round(r_sq2,4))
    
    return CAPM_table
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
#Carhart 4-Factor Model
def Carhart(data_portfolio, array_rf, data_market, data_smb, data_hml, data_umd):    
    #Carhart 4-Factor Model
    data_factors = pd.DataFrame((data_market-array_rf).values, 
                                columns = ['Rm-Rf'],
                                index = data_market.index)
    data_factors['SMB'] = data_smb.values
    data_factors['HML'] = data_hml.values
    data_factors['UMD'] = data_umd.values
    
    x = data_factors[['Rm-Rf','SMB', 'HML', 'UMD']].values.reshape(-1 ,4)
    y = data_portfolio - array_rf
    
    c = len(y.columns)
    
    #Regression of excess return of portfolio on the 3 factors
    model = [LinearRegression().fit(x, y.iloc[:,i]) for i in range(c)]
    r_sq = [model[i].score(x, y.iloc[:,i]) for i in range(c)]
    alpha = [model[i].intercept_ for i in range(c)]
    beta = [model[i].coef_[0] for i in range(c)]
    
    carhart_table = pd.DataFrame({'Alpha': alpha, 'Beta': beta, 'R2': r_sq}, 
                              index = y.columns)
    
    return carhart_table
#------------------------------------------------------------------------------
#data_portfolio = df of returns
#array_rf = (-1, 1) array of risk-free returns or scalar
#data_market = df of returns of the market
#data_smb = df of Small Minus Big
#data_hml = df of High Minus Low
#data_umd = df of Up Minus Down
#------------------------------------------------------------------------------    
#x0 =
#xt =
#num =
#------------------------------------------------------------------------------
#MM_table = MarketModel(data_portfolio, array_rf, data_market)
#print('------Market Model Regression------')
#print(MM_table)
#------------------------------------------------------------------------------
#CAPM_table = CAPM(data_portfolio, array_rf, data_market, MM_table, x0, xt, num)
#print('-------------CAPM Data-------------')
#print(CAPM_table)
#------------------------------------------------------------------------------    
#FF_table = FF3Factor(data_portfolio, array_rf, data_market, data_smb, data_hml)
#print('----------FF3Factor Data-----------')
#print(FF_table)
#------------------------------------------------------------------------------    
#carhart_table = Carhart(data_portfolio, array_rf, data_market, data_smb, data_hml, data_umd)
#print('-----------Carhart Data------------')
#print(carhart_table)
