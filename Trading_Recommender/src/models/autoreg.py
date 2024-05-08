import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from src.data.prepare_dataset import process_data
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import ar_select_order

def autoreg_forecast(args):
    '''    
    Generates a sequence of predicted stock values.

    Inputs
    ----------
    args: dictionary of dictionaries containing what follows.
    
        data_args: dictionary with strings as keys
            Data arguments. Must include:
            history: pandas DataFrame
                Contains the historical data
            order: positive integer
                Order of differencing required to make the series stationary
    
        model_args: defaultdict(int)
            Contains arguments for the model definition
            
        temporal_args: dictionary with strings as keys
            Temporal arguments. Must include:
            history: pandas DataFrame
                pd.DataFrame
                DataFrame with the timeseries data to learn from
            start_date: pandas Timestamp
                First date up to which the predictor will be fitted
            end_date: pandas Timestamp
                Last date that the predictions extend to
            horizon: positive integer
                Indicates up to how many days the forecast will extend to
        
        training_args: defaultdict(int) with strings as keys
            Training arguments

    Outputs
    ----------
    forecast:      Pandas series with the scaled and differenced forecast
    '''

    data_args = args['data_args']
    model_args = args['model_args']
    temporal_args = args['temporal_args']
    training_args = args['training_args']

    current_date = temporal_args['current_date']
    horizon = temporal_args['horizon']
    
    history = data_args['history']
    order = data_args['order']
    
    lags = model_args.get('lags', 1)
    maxlag = model_args.get('maxlag', 0)
     
    ## 1) Prepare the time series
    # Set up the time range
    mask = history.index < current_date
    to_fit = history[mask]
    # Differenciate and scale the time series
    to_fit, scaler, first_values = process_data(to_fit, order, 'pack')
    # Format the time index to be compatible with the model
    to_fit.index = pd.DatetimeIndex(to_fit.index).to_period('D')
    
    # Statistical tests to evaluate the stationarity
    # result = adfuller(to_fit)
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    
    ## 2) Fit the AutoReg to the historical data
    if maxlag > 0:
        sel = ar_select_order(to_fit, maxlag=maxlag)
        print(f'{sel.ar_lags = }')
        lags = sel.ar_lags
    predictor = AutoReg(to_fit, lags = lags)
    params = predictor.fit().params
    
    ## 3) Make a forecast 
    forecast = predictor.predict(params, 
                                 start = current_date + pd.Timedelta(1, 'D'), 
                                 end = current_date + pd.Timedelta(horizon, 'D'))
    
    ## 4) Process the result
    forecast.index = forecast.index.to_timestamp()
    forecast = forecast[forecast.index>current_date]
    forecast = forecast.to_frame()
    forecast.rename(columns={0: 'y'}, inplace=True)
    forecast.index.name = 'ds'
    forecast = process_data(forecast, order, direction='unpack', scaler=scaler)
    
    return forecast
