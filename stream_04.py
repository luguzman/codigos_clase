# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:37:27 2020

@author: Meva
"""

import numpy as np
# import pandas as pd
# import matplotlib as mpl
import scipy
import importlib
# import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

# import our own files and reload
import stream_functions
importlib.reload(stream_functions)


# input parameters
ric = '^VIX' # DBK.DE ^IXIC MXN=X ^STOXX ^S&P500 ^VIX
file_extension = 'csv'

x, x_str, t = stream_functions.load_timeseries(ric, file_extension)
stream_functions.plot_time_series_price(t, ric)


### Recycled code from stream_02.py ###

# compute "risk metrics"
x_size = len(x) # size of returns
x_mean = np.mean(x)
x_std = np.std(x) # volatility
x_skew = skew(x)
x_kurt = kurtosis(x) # excess kurtosis
x_sharpe = x_mean / x_std * np.sqrt(252) # annualised
x_var_95 = np.percentile(x,5)
x_cvar_95 = np.mean(x[x <= x_var_95])
jb = x_size/6*(x_skew**2 + 1/4*x_kurt**2)
p_value = 1 - chi2.cdf(jb, df=2)
is_normal = (p_value > 0.05) # equivalently jb < 6

# print metrics
round_digits = 4
str1 = 'mean ' + str(np.round(x_mean,round_digits))\
    + ' | std dev ' + str(np.round(x_std,round_digits))\
    + ' | skewness ' + str(np.round(x_skew,round_digits))\
    + ' | kurtosis ' + str(np.round(x_kurt,round_digits))\
    + ' | Sharpe ratio ' + str(np.round(x_sharpe,round_digits))
str2 = 'VaR 95% ' + str(np.round(x_var_95,round_digits))\
    + ' | CVaR 95% ' + str(np.round(x_cvar_95,round_digits))\
    + ' | jarque_bera ' + str(np.round(jb,round_digits))\
    + ' | p_value ' + str(np.round(p_value,round_digits))\
    + ' | is_normal ' + str(is_normal)
    
    
stream_functions.plot_histogram(x, x_str, str1, str2)