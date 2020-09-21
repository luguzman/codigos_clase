# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:59:17 2020

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

'''
Goal: create a normality test e.g. Jarque-Bera

step 1: generate random variables
step 2: visualise histogram
step 3: what is Jarque-Bera


'''

# generate random variable
x_size = 10**6
degrees_freedom = 500
type_random_variable = 'student' # normal exponential student

if type_random_variable == 'normal':
    x = np.random.standard_normal(size=x_size)
    x_str = type_random_variable
elif type_random_variable == 'exponential':
    x = np.random.standard_exponential(size=x_size)
    x_str = type_random_variable
elif type_random_variable == 'student':
    x = np.random.standard_t(size=x_size, df=degrees_freedom)
    x_str = type_random_variable + ' (df=' + str(degrees_freedom) + ')'


# compute "risk metrics"
x_mean = np.mean(x)
x_stdev = np.std(x)
x_skew = skew(x)
x_kurt = kurtosis(x) # excess kurtosis

# print metrics
print(x_str)
print('mean ' + str(x_mean))
print('std ' + str(x_stdev))
print('skewness ' + str(x_skew))
print('kurtosis ' + str(x_kurt))


# plot histogram
plt.figure()
plt.hist(x,bins=100)
plt.title('Histogram ' + x_str)
plt.show()


























