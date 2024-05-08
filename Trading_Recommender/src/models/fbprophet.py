import pandas as pd
from prophet import Prophet
from src.data.prepare_dataset import process_data

def fbprophet_forecast(args):
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
    
    include_dates = bool(model_args['include_dates'])
    n_changepoints = model_args.get('n_changepoints', 120)
     
    ## 1) Prepare the time series
    # Set up the time range
    mask = history.index < current_date
    to_fit = history[mask]
    # Differenciate and scale the time series
    to_fit, scaler, first_values = process_data(to_fit, order, 'pack')
    # Format the time index to be compatible with the model
    to_fit.reset_index(inplace=True)
    
    ## 2) Fit the fbprophet model to the historical data
    predictor = Prophet(growth='flat',
                        n_changepoints=n_changepoints)
    predictor.fit(to_fit[['y', 'ds']])
    
    ## 3) Make a forecast 
    freq_dict = {1:"d", 7:"w", 30:"m", 90: 'Q'}
    future = predictor.make_future_dataframe(periods = horizon, 
                                             freq = freq_dict[1], 
                                             include_history = include_dates)
    forecast = predictor.predict(future)
    
    ## 4) Process the result
    # Convert to compatible format
    forecast.rename(columns={'yhat': 'y'}, inplace = True)
    forecast = forecast[['y', 'ds']].set_index('ds')
    # Process
    forecast = process_data(forecast, order, direction='unpack', scaler=scaler) 
    forecast = forecast[forecast.index>current_date]
    
    return forecast