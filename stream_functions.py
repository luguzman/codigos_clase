# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:39:01 2020

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA


def print_number(n=5):
    print(n)


def load_timeseries(ric, file_extension='csv'):
    # get market data
    # remember to modify the path to match your own directory
    path = 'C:\\Users\Meva\\.spyder-py3\\data\\' + ric + '.' + file_extension
    if file_extension == 'csv':
        table_raw = pd.read_csv(path) # default csv
    else:
        table_raw = pd.read_excel(path)
    # create table of returns
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(table_raw['Date'], dayfirst=True)
    t['close'] = table_raw['Close']
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return_close'] = t['close']/t['close_previous'] - 1
    t = t.dropna()
    t = t.reset_index(drop=True)
    # input for Jarque-Bera test
    x = t['return_close'].values # returns as array
    x_str = 'Real returns ' + ric # label e.g. ric
    
    return x, x_str, t


def plot_time_series_price(t, ric):
    # plot timeseries of price
    plt.figure()
    plt.plot(t['date'],t['close'])
    plt.title('Time series real prices ' + ric)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    
    
def plot_histogram(x, x_str, plot_str, bins=100):
    # plot histogram
    plt.figure()
    plt.hist(x,bins)
    plt.title('Histogram ' + x_str)
    plt.xlabel(plot_str)
    plt.show()