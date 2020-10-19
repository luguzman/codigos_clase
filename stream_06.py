# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:14:02 2020

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
ric = 'BBVA.MC'
benchmark = '^STOXX'
# hedge_rics =  ['SAN.MC','REP.MC']
# hedge_rics = ['BBVA.MC','REP.MC']
# hedge_rics = ['SAN.MC','DBKGn.DE']
# hedge_rics = ['^FCHI','^GDAXI']
# hedge_rics = ['^STOXX','SAN.MC']
hedge_rics = ['^S&P500','^NASDAQ']
delta = 10 # mn USD

def compute_beta(ric, benchmark, bool_print=False):
    capm = stream_classes.capm_manager(ric, benchmark)
    capm.load_timeseries()
    capm.compute()
    if bool_print:
        print('------')
        print(capm)
    beta = capm.beta
    return beta

# portfolio beta
beta = compute_beta(ric, benchmark, bool_print=True)
beta_usd = beta*delta

# print input
print('------')
print('Input portfolio:')
print('Delta mnUSD for ' + ric + ' is ' + str(delta))
print('Beta for ' + ric + ' vs ' + benchmark + ' is ' + str(beta))
print('Beta mnUSD for ' + ric + ' vs ' + benchmark + ' is ' + str(beta_usd))

# compute betas for the hedges
shape = [len(hedge_rics),1]
betas = np.zeros(shape)
counter = 0
print('------')
print('Input hedges:')
for hedge_ric in hedge_rics:
    beta = compute_beta(hedge_ric, benchmark)
    print('Beta for hedge[' + str(counter) + '] = ' + hedge_ric + ' vs ' + benchmark + ' is ' + str(beta))
    betas[counter] = beta
    counter += 1
    
# exact solution using matrix algebra
deltas = np.ones(shape)
targets = -np.array([[delta],[beta_usd]])
mtx = np.transpose(np.column_stack((deltas,betas)))
optimal_hedge = np.linalg.inv(mtx).dot(targets)
hedge_delta = np.sum(optimal_hedge)
hedge_beta_usd = np.transpose(betas).dot(optimal_hedge).item()

# print result
print('------')
print('Optimisation result')
print('------')
print('Delta: ' + str(delta))
print('Beta USD: ' + str(beta_usd))
print('------')
print('Hedge delta: ' + str(hedge_delta))
print('Hedge beta: ' + str(hedge_beta_usd))
print('------')
print('Betas for the hedge:')
print(betas)
print('------')
print('Optimal hedge:')
print(optimal_hedge)
print('------')
