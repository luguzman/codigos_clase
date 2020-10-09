# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:17:25 2020

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress

# import our own files and reload
import stream_functions
importlib.reload(stream_functions)
import stream_classes
importlib.reload(stream_classes)

# input parameters
ric = '^STOXX' # MT.AS SAN.MC BBVA.MC REP.MC VWS.CO EQNR.OL MXNUSD=X ^VIX
benchmark = '^S&P500' # ^STOXX50E ^STOXX ^S&P500 ^NASDAQ ^FCHI ^GDAXI
file_extension = 'csv'
nb_decimals = 4

# loading data from csv or Excel file
x1, str1, t1 = stream_functions.load_timeseries(ric)
x2, str2, t2 = stream_functions.load_timeseries(benchmark)

# synchronize timestamps
timestamp1 = list(t1['date'].values)
timestamp2 = list(t2['date'].values)
timestamps = list(set(timestamp1) & set(timestamp2))

# synchronised time series for x1 or ric
t1_sync = t1[t1['date'].isin(timestamps)]
t1_sync.sort_values(by='date', ascending=True)
t1_sync = t1_sync.reset_index(drop=True)

# synchronised time series for x2 or benchmark
t2_sync = t2[t2['date'].isin(timestamps)]
t2_sync.sort_values(by='date', ascending=True)
t2_sync = t2_sync.reset_index(drop=True)

# table of returns for ric and benchmark
t = pd.DataFrame()
t['date'] = t1_sync['date']
t['price_1'] = t1_sync['close']
t['price_2'] = t2_sync['close']
t['return_1'] = t1_sync['return_close']
t['return_2'] = t2_sync['return_close']

# compute vectors of returns
y = t['return_1'].values
x = t['return_2'].values

# linear regression of ric with respect to benchmark
slope, intercept, r_value, p_value, std_err = linregress(x,y)
slope = np.round(slope, nb_decimals)
intercept = np.round(intercept, nb_decimals)
p_value = np.round(p_value, nb_decimals) 
null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
r_value = np.round(r_value, nb_decimals) # correlation coefficient
r_squared = np.round(r_value**2, nb_decimals) # pct of variance of y explained by x
predictor_linreg = slope*x + intercept

# scatterplot of returns
str_title = 'Scatterplot of returns' + '\n'\
    + 'Linear regression | ric ' + ric\
    + ' | benchmark ' + benchmark + '\n'\
    + 'alpha (intercept) ' + str(intercept)\
    + ' | beta (slope) ' + str(slope) + '\n'\
    + 'p-value ' + str(p_value)\
    + ' | null hypothesis ' + str(null_hypothesis) + '\n'\
    + 'r-value ' + str(r_value)\
    + ' | r-squared ' + str(r_squared)
plt.figure()
plt.title(str_title)
plt.scatter(x,y)
plt.plot(x, predictor_linreg, color='green')
plt.ylabel(ric)
plt.xlabel(benchmark)
plt.grid()
plt.show()
