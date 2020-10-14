# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:10:59 2020

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


class jarque_bera_test():
    
    def __init__(self, ric):
        self.ric = ric
        self.returns = []
        self.t = pd.DataFrame()
        self.size = 0
        self.str_name = None
        self.mean = 0.0
        self.std = 0.0
        self.skew = 0.0
        self.kurt = 0.0
        self.median = 0.0
        self.var_95 = 0.0
        self.cvar_95 = 0.0
        self.sharpe = 0.0
        self.jarque_bera = 0.0
        self.p_value = 0.0
        self.is_normal = 0.0
        
        
    def __str__(self):
        str_self = self.str_name + ' | size ' + str(self.size) + '\n' + self.plot_str()
        return str_self


    def load_timeseries(self):
        self.returns, self.str_name, self.t = stream_functions.load_timeseries(self.ric)
        self.size = len(self.returns)
        
        
    def generate_ramdom_vector(self, type_random_variable, size=10**6, degrees_freedom=None):
        # type_random_variable normal exponential student
        if type_random_variable == 'normal':
            self.returns = np.random.standard_normal(size)
            self.str_name = 'Standard Normal RV'
        elif type_random_variable == 'exponential':
            self.returns = np.random.standard_exponential(size)
            self.str_name = 'Exponential RV'
        elif type_random_variable == 'student':
            if degrees_freedom == None:
                degrees_freedom = 750 # borderline for Jarque-Bera with 10**6 samples
            self.returns = np.random.standard_t(df=degrees_freedom,size=size)
            self.str_name = 'Student RV (df = ' + str(degrees_freedom) + ')'
        elif type_random_variable == 'chi-squared':
            if degrees_freedom == None:
                degrees_freedom = 2 # Jarque-Bera test uses 2 degrees of freedom
            self.returns = np.random.chisquare(df=degrees_freedom,size=size)
            self.str_name = 'Chi-squared RV (df = ' + str(degrees_freedom) + ')'


    def compute(self):
        self.mean = np.mean(self.returns)
        self.std = np.std(self.returns) # volatility
        self.skew = skew(self.returns)
        self.kurt = kurtosis(self.returns) # excess kurtosis
        self.sharpe = self.mean / self.std * np.sqrt(252) # annualised
        self.median = np.median(self.returns)
        self.var_95 = np.percentile(self.returns,5)
        self.cvar_95 = np.mean(self.returns[self.returns <= self.var_95])
        self.jarque_bera = self.size/6*(self.skew**2 + 1/4*self.kurt**2)
        self.p_value = 1 - chi2.cdf(self.jarque_bera, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6


    def plot_str(self):
        nb_decimals = 4
        plot_str = 'mean ' + str(np.round(self.mean,nb_decimals))\
            + ' | median ' + str(np.round(self.median,nb_decimals))\
            + ' | std dev ' + str(np.round(self.std,nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew,nb_decimals)) + '\n'\
            + 'kurtosis ' + str(np.round(self.kurt,nb_decimals))\
            + ' | Sharpe ratio ' + str(np.round(self.sharpe,nb_decimals))\
            + ' | VaR 95% ' + str(np.round(self.var_95,nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95,nb_decimals)) + '\n'\
            + 'Jarque Bera ' + str(np.round(self.jarque_bera,nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value,nb_decimals))\
            + ' | is normal ' + str(self.is_normal)
        return plot_str
    
    
class capm_manager():
    
    def __init__(self, ric, benchmark):
        self.ric = ric
        self.benchmark = benchmark
        self.x = []
        self.y = []
        self.t = pd.DataFrame()
        self.alpha = None
        self.beta = None
        self.p_value = None
        self.null_hypothesis = False
        self.r_value = None
        self.r_squared = None
        self.predictor_linreg = []
        
        
    def __str__(self):
        # str_self = '__str__ not yet defined, next course please'
        str_self = 'Linear regression | ric ' + self.ric\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' | beta (slope) ' + str(self.beta) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + ' | null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'r-value ' + str(self.r_value)\
            + ' | r-squared ' + str(self.r_squared)
        return str_self
        
    
    def load_timeseries(self):
        # load timeseries and synchronise them
        self.x, self.y, self.t = stream_functions.synchronise_timeseries(self.ric, self.benchmark)
    
    
    def compute(self):
        # linear regression of ric with respect to benchmark
        nb_decimals = 4
        slope, intercept, r_value, p_value, std_err = linregress(self.x,self.y)
        self.beta = np.round(slope, nb_decimals)
        self.alpha = np.round(intercept, nb_decimals)
        self.p_value = np.round(p_value, nb_decimals) 
        self.null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
        self.r_value = np.round(r_value, nb_decimals) # correlation coefficient
        self.r_squared = np.round(r_value**2, nb_decimals) # pct of variance of y explained by x
        self.predictor_linreg = self.alpha + self.beta*self.x
        
        
    def scatterplot(self):
        # scatterplot of returns
        str_title = 'Scatterplot of returns' + '\n' + self.__str__()
        plt.figure()
        plt.title(str_title)
        plt.scatter(self.x,self.y)
        plt.plot(self.x, self.predictor_linreg, color='green')
        plt.ylabel(self.ric)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()
        
        
    def plot_normalised(self):
        # plot 2 timeseries normalised at 100
        price_ric = self.t['price_1']
        price_benchmark = self.t['price_2'] 
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices | normalised at 100')
        plt.xlabel('Time')
        plt.ylabel('Normalised prices')
        price_ric = 100 * price_ric / price_ric[0]
        price_benchmark = 100 * price_benchmark / price_benchmark[0]
        plt.plot(price_ric, color='blue', label=self.ric)
        plt.plot(price_benchmark, color='red', label=self.benchmark)
        plt.legend(loc=0)
        plt.grid()
        plt.show()
        
        
    def plot_dual_axes(self):
        # plot 2 timeseries with 2 vertical axes
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        ax1 = self.t['price_1'].plot(color='blue', grid=True, label=self.ric)
        ax2 = self.t['price_2'].plot(color='red', grid=True, secondary_y=True, label=self.benchmark)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
        