import os
try: 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    assert dir_path == os.getcwd()
except AssertionError:
    print("Changing working directory to that of the ValueInvestor project")
    os.chdir(dir_path)

import yfinance as yf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.diagnostics import cross_validation
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller
from numpy import log
pd.plotting.register_matplotlib_converters()

print('setup file was run')

'''
            n_a: Integer, number of hidden units in the LSTM
            seq_len: Integer, sequence length, window size traversing the series
            include_dates: Boolean, whether to include the dates for every time point.
            difference: Boolean, whether to difference the time series or not.
                        No action is performed by default.
'''
