import pandas as pd
from src.data.prepare_dataset import process_data
from src.models.LSTM_files.LSTM_fit import fit_LSTM
from src.models.LSTM_files.LSTM_predict import predict_LSTM

def LSTM_forecast(args):
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
            
        model_args: defaultdict(int) with strings as keys
            Model arguments. May include:
            seq_len: Positive integer
                Sequence length / number of timesteps in each LSTM training example
            n_features: Positive integer
                Number of features to predict in the data
            n_a: Positive integer
                Number of LSTM units
            learning_rate: float
                Learning rate during LSTM training
            loss: string
                Name of the loss function used during LSTM training
            stateful_training: boolean
                Whether the LSTM training will be in stateful mode
            stateful_inference: boolean
                Whether the LSTM inference will be in stateful mode
            
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
            Training arguments. May include:
            epochs: Positive integer
                Number of epochs when a keras model fits to the data
            batch_size: Positive integer
                batch size during training
            shuffle: Boolean
                Whether to shuffle the dataset during training
            verbose: Integer (0 or 1)
                verbose parameter during LSTM fitting

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
    
    ## 2) Fit the LSTM to the historical data
    model, training_data, val_score = fit_LSTM(to_fit, model_args, training_args)
    
    ## 3) Make a forecast 
    forecast = predict_LSTM(to_fit,
                            model,
                            model_args,
                            current_date)
    
    ## 4) Unscale and integrate the forecast
    forecast = process_data(forecast, order, 'unpack', scaler, first_values)
    
    return forecast