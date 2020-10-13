# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:37:27 2020

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

# import our own files and reload
import stream_functions
importlib.reload(stream_functions)
import stream_classes
importlib.reload(stream_classes)

# input parameters
ric = 'SGREN.MC' # DBK.DE ^IXIC MXN=X ^STOXX ^S&P500 ^VIX
file_extension = 'csv' # csv o Excel extension

# load timeseries
x, x_str, t = stream_functions.load_timeseries(ric)

# compute risk metrics in class jarque_bera_test
jb = stream_classes.jarque_bera_test(x, x_str)
jb.compute()
print(jb)

# plots
stream_functions.plot_timeseries_price(t, ric)
stream_functions.plot_histogram(x, x_str, jb.plot_str())
