import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from src.data.prepare_dataset import process_data

def SimpleExpSmoothing_forecast(args):
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
     
    ## 1) Prepare the time series
    # Set up the time range
    mask = history.index < current_date
    to_fit = history[mask]
    # Differenciate and scale the time series
    to_fit, scaler, first_values = process_data(to_fit, order, 'pack')
    # Format the time index to be compatible with the model
    # to_fit.index = pd.DatetimeIndex(to_fit.index).to_period('D')
    
    ## 2) Fit the SimpleExpSmoothing to the historical data
    predictor = SimpleExpSmoothing(to_fit)
    params = predictor.fit().params
    
    ## 3) Make a forecast 
    forecast = predictor.predict(params,
                                 start = current_date + pd.Timedelta(1, 'D'),
                                 end = current_date + pd.Timedelta(horizon, 'D'))
    
    ## 4) Process the result
    index = [current_date + pd.Timedelta(i+1, 'D') for i in range(len(forecast))]
    forecast = pd.DataFrame(data=forecast, index=index, columns = ['y'])
    forecast.index.name = 'ds'
    forecast = process_data(forecast, order, direction='unpack', scaler=scaler)
    forecast = forecast[forecast.index > current_date]
    
    return forecast