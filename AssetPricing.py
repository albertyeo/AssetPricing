#Multi-Period Asset Pricing

'''
Monte Carlo simulation for consumption growth, taking into account the
possibility of rare disasters, to:
1. Calculate and plot pricing kernel's standard deviation and mean ratio 
against gamma (power utility function) to find the smallest value of gamma for 
which Hansen-Jagannathan Bound is satisfied
2. Calculate price-dividend ratio and equity premium and plot these values 
against gamma
'''
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
def MPAP(simulation):
    #Value of consumption growth
    list_g = []
    for i in range(simulation):
        eps = np.random.normal()
        rand_nu = np.random.uniform()
        if rand_nu <= 0.017:
            nu = np.log(0.65)
        else:
            nu = 0 #check the function of nu
        rand_g = np.exp(0.02 + 0.02 * eps + nu)
        list_g.append(rand_g)
    g = np.array(list_g)
    
    #Hansen-Jagannathan bound
    gamma = np.linspace(1, 4, 100)
    mean_m = []
    std_m = []
    list_value = []
    for i in gamma:
        m = 0.99 * g ** (-i)
        mean_m.append(np.mean(m))
        std_m.append(np.std(m))
        if np.std(m)/np.mean(m) > 0.4:
            list_value.append(i)
        
    ratio_m = np.array(std_m)/np.array(mean_m)
    
    plt.plot(gamma, ratio_m)
    plt.ylabel('SD(M)/E(M)')
    plt.xlabel('Gamma')
    plt.show()
    print('The smallest value of gamma for which SD(M)/E(M) > 0.4 is ', np.min(list_value))
    #check the value for print (0.4?)
    
    #Price-Dividend ratio
    gamma_2 = np.linspace(1, 7, 100)
    list_p1_d = []
    mean_m2 = []
    for i in gamma_2:
        p1_d = 0.99 * g ** (1 - i)
        list_p1_d.append(np.mean(p1_d))
        m2 = 0.99 * g ** (-i)
        mean_m2.append(np.mean(m2))
    
    plt.plot(gamma_2, list_p1_d)
    plt.ylabel('P1/D')
    plt.xlabel('Gamma')
    plt.show()

    #Equity premium
    r_m = (1 / np.array(list_p1_d)) * np.mean(g)
    r_f = 1 / np.array(mean_m2)
    eq_premium = r_m - r_f
    
    plt.plot(gamma_2, eq_premium)
    plt.ylabel('Equity Premium')
    plt.xlabel('Gamma')
    plt.show()
#------------------------------------------------------------------------------
#MPAP(simulation)
