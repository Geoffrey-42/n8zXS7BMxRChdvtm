import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from src.data.prepare_dataset import process_data

def arima_forecast(args):
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
            Model arguments. Must include:
            arima_order: tuple
                (p, d, q) from ARIMA
                with d = 0 because:
                The series was already differenced {data_args['order']} times
            
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
    
    arima_order = model_args['arima_order']
    autosearch_arima = bool(model_args['autosearch_arima'])
     
    ## 1) Prepare the time series
    # Set up the time range
    mask = history.index < current_date
    to_fit = history[mask]
    # Differenciate and scale the time series
    to_fit, scaler, first_values = process_data(to_fit, order, 'pack')
    
    ## 2) Fit the ARIMA to the historical data
    if autosearch_arima:
        from pmdarima.arima import auto_arima
        auto_search = auto_arima(to_fit, seasonal=False, trace=True)
        arima_order = auto_search.order
    model = ARIMA(to_fit, 
                  order = arima_order)
    predictor = model.fit()
    
    ## 3) Make a forecast 
    forecast = predictor.forecast(steps = horizon)
    
    ## 4) Process the result
    forecast = forecast.to_frame()
    forecast.rename(columns={'predicted_mean': 'y'}, inplace=True)
    forecast.index.name = 'ds'
    forecast = forecast[forecast.index>current_date]
    forecast = process_data(forecast, order, direction='unpack', scaler=scaler)
    
    return forecast