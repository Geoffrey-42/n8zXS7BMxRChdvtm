## Import classes and functions from internal modules
from src.data.extract_dataset import extract_financial_data
from src.data.debug_dataset import generate_debug_time_series
from src.forecast.forecaster import forecaster
from src.forecast.recommender import recommender
from src.tuning.optuna_tuning import optuna_search


## Import external libraries
import os
import pandas as pd

pd.plotting.register_matplotlib_converters()

import optuna_dashboard
from optuna.storages import JournalStorage, JournalFileStorage

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')


### Get the data
hist = extract_financial_data(data_dir=data_dir)
debug_period = 120
debug_series = generate_debug_time_series(debug_period)

# Choose data
symbols = ["BTC-USD",
           "GOOG",
           "MSFT",
           "KCHOL.IS",
           "BEEF3.SA",
           "PAM",
           "CMTOY",
           "IMP.JO"]
symbol = symbols[2]
history = hist[symbol][['y']]

debug = False
if debug:
    history = debug_series
    symbol = 'Cosinus'

# Choose the forecasting horizon
if debug:
    horizon = 14 # in days
else:
    horizon = 14

### Choose the model
predictor_names = ['LSTM',
                   'fbprophet',
                   'AutoReg',
                   'ARIMA',
                   'SARIMAX',
                   'SimpleExpSmoothing']

predictor_name = predictor_names[0]

## Set the model parameters
model_args = dict()
if predictor_name == 'LSTM':
    model_args = {'seq_len': 120,
                  'n_features': 1,
                  'learning_rate': 0.0001,
                  'loss': 'mse',
                  'n_a': 32,
                  'dropout': 0.05,
                  'stateful_training': False,
                  'stateful_inference': False,
                  'horizon': horizon}
    if debug:
        model_args['learning_rate'] = 0.0005
        model_args['seq_len'] = 30
        model_args['n_a'] = 16
        model_args['dropout'] = 0
        model_args['loss'] = 'mse'
elif predictor_name == 'ARIMA':
    if debug:
        model_args['autosearch_arima'] = False
        model_args['arima_order'] = (6, 0, 1) # (p, d, q)
    else:
        model_args['autosearch_arima'] = False
        model_args['arima_order'] = (6, 0, 1) # (p, d, q)
    # Note: d=0 because the time series will already be differenced if needed
elif predictor_name == 'SARIMAX':
    if debug:
        model_args['autosearch_arima'] = False
        model_args['arima_order'] = (0, 0, 0) # (p, d, q)
        model_args['seasonal_order'] = (0, 1, 0, debug_period) # (P, D, Q, m)   
    else:
        model_args['autosearch_arima'] = False
        model_args['arima_order'] = (6, 0, 1) # (p, d, q)
        model_args['seasonal_order'] = (1, 1, 0, 30) # (P, D, Q, m)
elif predictor_name == 'AutoReg':
    model_args['lags'] = 120
    if debug:
        model_args['maxlag'] = debug_period
elif predictor_name == 'fbprophet':
    if debug:
        model_args['n_changepoints'] = debug_period


## Set the training parameters
training_args = {'epochs': 100,
                 'batch_size': 32,
                 'shuffle': False,
                 'verbose': 1}
if debug:
    training_args = {'epochs': 100,
                     'batch_size': 32,
                     'shuffle': False,
                     'verbose': 1}


### Setting up the simulation parameters

## Set the data parameters
order = 0
# Note: After running the Augmented Dickey Fuller Test, it was found that
# these time series should be differenced 1 time to become stationary.
# However, after experimenting, it was found that the LSTM performed better 
# when the series were not differenced. While order should be set to 1 for 
# linear models like ARIMA, it is recommended to keep it to 0 for LSTM

if debug:
    order = 0 # The debug series is already stationary

data_args = {'history': history,
             'order': order,
             'symbol': symbol,
             'plot_start_date': pd.to_datetime('2023-01-01')}


## Set the temporal parameters
start_date = pd.to_datetime('2023-07-01')
end_date = pd.to_datetime('2024-01-01')
temporal_args = {'start_date': start_date,
                 'end_date': end_date,
                 'horizon': horizon}


## Set the trading parameters
initial_stock = 1
max_trade = 1
intensity = 3 # Price variation by 1/intensity results in trading max_trade
min_rate = 0.003 # Minimum daily rate of relative price change to trigger trading action
trading_args = {'initial_stock': initial_stock,
                'max_trade': max_trade,
                'intensity': intensity,
                'min_delta': min_rate*horizon}


## Defining an optuna study (Temporal parameters should define the validation set)
hyperparameter_search = False
if hyperparameter_search:
    if debug:
        study_name = f'cosinus_LSTM_ahead={horizon}_period={debug_period}_order=1'
        storage_name = 'debug_LSTM'
    else:
        study_name = f'{symbol}_LSTM_ahead={horizon}'
        storage_name = 'tuning_LSTM'
    # Define the study storage method
    storage = JournalStorage(JournalFileStorage(f"src/tuning/{storage_name}.log"))
    # Pack the arguments
    args = (model_args, data_args, temporal_args, training_args, trading_args)
    # Launch the search
    n_a, learning_rate, seq_len = optuna_search(20,
                                                storage,
                                                study_name,
                                                args,
                                                na_range = (32, 256),
                                                lr_range = (0.0001, 0.01),
                                                seq_len_range = (60, 240),
                                                dropout_range = (0, 0.4)
                                                )
    # Assign the results
    model_args['n_a'] = n_a
    model_args['learning_rate'] = learning_rate
    model_args['seq_len'] = seq_len
    model_args['dropout'] = dropout
    # Run the dashboard
    optuna_dashboard.run_server(storage)


### Calling the forecaster and recommender objects

## Set the temporal parameters (Should define the test set)
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-04-26')
temporal_args = {'start_date': start_date,
                 'end_date': end_date,
                 'horizon': horizon}

## Create a forecaster object
clairvoyant = forecaster(predictor_name,
                         model_args)

## Create a recommender object
recommend = recommender(oracle = clairvoyant,
                        trading_args = trading_args)


### Simulate forecasting and recommendations
recommend(data_args,
          temporal_args,
          training_args) # performs the recommendation