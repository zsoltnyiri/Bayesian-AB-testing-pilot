# Bayesian AB testing based on https://www.evanmiller.org/bayesian-ab-testing.html

from math import lgamma
from numba import jit
import numpy as np
from scipy.stats import beta
from calc_prob import calc_prob_between
import matplotlib.pyplot as plt

# defining the functions used
@jit
def h(a, b, c, d):
    num = lgamma(a + c) + lgamma(b + d) + lgamma(a + b) + lgamma(c + d)
    den = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
    return np.exp(num - den)

@jit
def g0(a, b, c):    
    return np.exp(lgamma(a + b) + lgamma(a + c) - (lgamma(a + b + c) + lgamma(a)))

@jit
def hiter(a, b, c, d):
    while d > 1:
        d -= 1
        yield h(a, b, c, d) / d

def g(a, b, c, d):
    return g0(a, b, c) + sum(hiter(a, b, c, d))

def calc_prob_between(beta1, beta2):
    return g(beta1.args[0], beta1.args[1], beta2.args[0], beta2.args[1])

def calc_beta_mode(a, b):
    '''this function calculate the mode (peak) of the Beta distribution'''
    return (a-1)/(a+b-2)

def plot(betas, names, linf=0, lsup=0.01):
    '''this function plots the Beta distribution'''
    x=np.linspace(linf,lsup, 100)
    for f, name in zip(betas,names) :
        y=f.pdf(x) #this for calculate the value for the PDF at the specified x-points
        y_mode=calc_beta_mode(f.args[0], f.args[1])
        y_var=f.var() # the variance of the Beta distribution
        plt.plot(x,y, label=f" version: {name}, metric: {round(y_mode * 100, 2)} $\pm$ {y_var:0.1E}")
        plt.yticks([])
    plt.legend()
    plt.show()

# Setting parameters
metric = 'conversion rate'

name_test, imps_test, convs_test = 'version A',  573, 100
name_ctrl, imps_ctrl, convs_ctrl = 'version B', 567, 86

# Create the Beta functions for the two sets
a_C, b_C = convs_ctrl + 1, imps_ctrl-convs_ctrl + 1
beta_C = beta(a_C, b_C)
a_T, b_T = convs_test + 1, imps_test-convs_test + 1
beta_T = beta(a_T, b_T)

# Calculating the lift
lift = (beta_T.mean()-beta_C.mean())/beta_C.mean()

# Calculating the probability for Test to be better than Control
prob = calc_prob_between(beta_T, beta_C)

print(f'Parameters: \n Metric: {metric} \n  - Version A: {name_ctrl}  - Observed users: {imps_ctrl} - Converted: {convs_ctrl} - CR: {round((convs_ctrl/imps_ctrl*100), 2)}% \n  - Version B: {name_test}  - Observed users: {imps_test} - Converted: {convs_test} - CR: {round((convs_test/imps_test*100), 2)}% \n')
print(f" Result: \n  - The version {name_test} lifts - {metric} - by {lift * 100:2.2f}% with {prob * 100:2.2f}% probability compared to the version \n {name_ctrl}.")

plot([beta_C, beta_T], names=[name_ctrl, name_test], linf = (convs_ctrl/imps_ctrl) - 0.04, lsup = (convs_test/imps_test) + 0.04)