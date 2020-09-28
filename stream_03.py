# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:10:54 2020

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2


# input
ric = '^IXIC' # DBK.DE ^IXIC MXN=X

# get market data
# remember to modify the path to match your own directory
path = 'C:\\Users\Meva\\.spyder-py3\\data\\' + ric + '.csv' 
table_raw = pd.read_csv(path)

# create table of returns
t = pd.DataFrame()
t['date'] = pd.to_datetime(table_raw['Date'], dayfirst=True)
t['close'] = table_raw['Close']
t.sort_values(by='date', ascending=True)
t['close_previous'] = t['close'].shift(1)
t['return_close'] = t['close']/t['close_previous'] - 1
t = t.dropna()
t = t.reset_index(drop=True)

# plot timeseries of prices
plt.figure()
plt.plot(t['date'],t['close'])
plt.title('Time series real prices ' + ric)
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()


# input for Jarque-Bera test
x = t['return_close'].values # returns as array
x_str = 'Real returns ' + ric # label e.g. ric
x_size = len(x) # size of returns


### Recycled code from stream_02 ###

# compute "risk metrics"
x_mean = np.mean(x)
x_stdev = np.std(x) # volatility
x_skew = skew(x)
x_kurt = kurtosis(x) # excess kurtosis
x_sharpe = x_mean / x_stdev * np.sqrt(252) # annualised
x_var_95 = np.percentile(x,5)
x_cvar_95 = np.mean(x[x <= x_var_95])
jb = x_size/6*(x_skew**2 + 1/4*x_kurt**2)
p_value = 1 - chi2.cdf(jb, df=2)
is_normal = (p_value > 0.05) # equivalently jb < 6

# print metrics
print(x_str)
print('mean ' + str(x_mean))
print('std ' + str(x_stdev))
print('skewness ' + str(x_skew))
print('kurtosis ' + str(x_kurt))
print('Sharpe ' + str(x_sharpe))
print('VaR 95% ' + str(x_var_95))
print('CVaR 95% ' + str(x_cvar_95))
print('Jarque-Bera ' + str(jb))
print('p-value ' + str(p_value))
print('is normal ' + str(is_normal))
    
# plot histogram
plt.figure()
plt.hist(x,bins=100)
plt.title('Histogram ' + x_str)
plt.show()