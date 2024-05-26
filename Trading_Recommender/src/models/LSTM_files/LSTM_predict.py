import numpy as np
import pandas as pd

def predict_LSTM(time_series,
                 model,
                 model_args,
                 current_date,
                 horizon):
    '''    
    Generates a sequence of predicted stock values.

    Inputs
    ----------
    training_data: tuple containing two numpy arrays
    model:         Keras LSTM trained model
    model_args:    dictionary, model arguments
    current_date:  datetime, Date after which predictions will be made

    Outputs
    ----------
    forecast:      Pandas series with the scaled and differenced forecast
    '''

    ## 1) Instantiate the predictor model
    predictor = predictor_model(model,
                                model_args)
    
    ## 2) Prepare the input
    predict_input = prepare_input(time_series,
                                  model_args)

    ## 3) Make a prediction
    prediction = predictor([predict_input])
    
    ## 4) Extract the dates and values and store them in lists
    forecast = form_dataframe_from_prediction(prediction,
                                              current_date,
                                              horizon)
    
    return forecast

def predictor_model(model,
                    model_args):
    '''    
    This function exists in order to allow the usage of a stateful predictive 
    LSTM model using weights obtained under non stateful training.
    
    The goal is to compare the performance of the following approaches:
        LSTM model trained under stateful=False, predict under stateful=False
        LSTM model trained under stateful=False, predict under stateful=True
        LSTM model trained under stateful=True, predict under stateful=True
        
    In the case where stateful differs between training and predicting, a new 
    model is created for predicting and weights are transferred from the original.
    Else, the same model is returned.

    Inputs:
    model:      Keras LSTM trained model
    model_args: Dictionary, model arguments

    Outputs:
    predictive_model:  The predictive model
    '''
    from src.models.LSTM_files.LSTM_model import LSTM_model
    if model_args['stateful_training'] == model_args['stateful_inference']:
        predictive_model = model # same model
    else: # Transfer weights from non stateful model to stateful model
        predictive_model = LSTM_model(model_args, training=False)
        predictive_model.layers[0].set_weights(model.layers[0].get_weights())
        predictive_model.layers[2].set_weights(model.layers[2].get_weights())
    
    return predictive_model


def prepare_input(time_series,
                  model_args = {}):
    '''
    Prepare the initial input for the LSTM model.

    Inputs
    ----------
    time_series: Pandas DataFrame with dates as indices

    model_args: dictionary with the requested sequence length and/or n_features
    
    Outputs
    ----------
    x: A numpy array of dimension (1, n_timesteps, n_features).
       The input sequence for the next prediction.
    '''
    
    seq_len = model_args.get('seq_len', 5)
    n_features = time_series.shape[1]-1
    
    x = np.zeros((1, seq_len, n_features))
    
    x[0, :, :] = time_series.iloc[-seq_len:, :n_features].values
    
    return x


def form_dataframe_from_prediction(prediction,
                                   current_date,
                                   horizon):
    '''    
    Generates a dataframe out of a prediction.

    Inputs
    ----------
    prediction:   list of 1 element arrays, sequence of predictions
    model_args:   dictionary, model arguments
    current_date: datetime, Date after which predictions will be made

    Outputs
    ----------
    forecast:      Pandas series with the scaled and differenced forecast
    '''
    # Date
    next_date = current_date + pd.Timedelta(horizon, 'D')
    if next_date.dayofweek >= 5: # no data on weekends
        next_date = current_date + pd.Timedelta(horizon+2, 'D')
    date = [next_date]
        
    # Predicted value
    y = prediction[0]
        
    # Pandas time series containing the predictions
    forecast = pd.DataFrame(data = {'Return': y, 'Date': date}).set_index('Date')
    
    return forecast

# Testing prepare_initial_input
if (__name__ == "__main__"):
    print('\nTesting prepare_input')
    
    dates = pd.date_range(start='2020-01-01', 
                          end='2024-05-01', 
                          freq='B', # 'B' pour les jours ouvrés
                          name='Date')
    
    Close = [i for i in range(len(dates))]
    
    series = pd.DataFrame({'Close': Close}, index = dates)

    x = prepare_input(series,
                      {'seq_len': 5})
    
    print(f'\n{series.tail() = }')
    print(f'\n{x = }')
    print(f'\n{x.shape = }\n')