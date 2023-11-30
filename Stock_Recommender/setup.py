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

import tensorflow as tf
from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import optuna
from optuna.samplers import TPESampler

from tensorflow.keras.layers import LSTM, Dense, Reshape, RepeatVector, Concatenate, Flatten, Dropout
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanSquaredError
from tensorflow import transpose, reshape, make_tensor_proto, make_ndarray, convert_to_tensor, float32

print('setup file was run')

