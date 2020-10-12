# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:16:10 2020

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
ric = '^VIX' # MT.AS SAN.MC BBVA.MC REP.MC VWS.CO EQNR.OL MXNUSD=X ^VIX
benchmark = '^S&P500' # ^STOXX50E ^STOXX ^S&P500 ^NASDAQ ^FCHI ^GDAXI
file_extension = 'csv'
nb_decimals = 4

capm = stream_classes.capm_manager(ric, benchmark)
capm.load_timeseries()
capm.compute()
capm.scatterplot()
print(capm)
capm.__str__()

