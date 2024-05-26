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
            features: pandas DataFrame
                Time series engineered features related to the stock forecasted
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
    
    features = data_args['features']
     
    ## 1) Prepare the time series
    # Set up the time range
    mask = features.index < current_date
    features = features[mask]

    ## 2) Fit the LSTM to the historical data
    model, training_data, train_score = fit_LSTM(features, model_args, training_args)
    # print('Train MSE = %s\n Train MAE = %s'%tuple(train_score))
    # print(f'{model.summary()}')
    
    ## 3) Make a forecast 
    forecast = predict_LSTM(features,
                            model,
                            model_args,
                            current_date,
                            horizon)
    
    return forecast