import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from src.models.LSTM_files.LSTM_model import LSTM_model

def fit_LSTM(time_series,
             model_args,
             training_args):
    '''
    Fits the forecaster's model.
    
    Inputs
    ----------
    time_series: Pandas DataFrame with dates as indices
                 Time series with data the model will be fitted to.
                 It is assumed to be scaled and stationary.
            
    Outputs
    ----------
    LSTM_layer: A trained LSTM layer
    dense: A trained Dense layer
    scaler: A Scaler that has been fitted to the data
    x0: Array, sequence that follows the last labeled sequence.
        Will be used as an input to predict_LSTM
    a0: Array, initial LSTM hidden state
    c0: Array, initial LSTM cell state
    '''
    
    ## 1) Data Preparation for LSTM
    
    X, Y = prepare_data_for_LSTM(time_series,
                                 model_args)
    
    ## 2) Create the model in training mode
    model = LSTM_model(model_args, training = True)
    
    ## 3) Configure the training arguments
    training_args = configure_training_args(training_args)
    
    ## 4) Fit the model to the data
    val_score = []
    if model_args['autofit']:
        tscv = TimeSeriesSplit(n_splits=3)
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            Y_train, Y_val = Y[train_index], Y[val_index]
            # loops > 1 and epochs=1 when stateful=True
            for loop in range(training_args['loops']):
                model.layers[0].reset_states()
                model.fit(X_train, 
                          Y_train, 
                          epochs = training_args['epochs'], 
                          batch_size = training_args['batch_size'],
                          validation_data = (X_val, Y_val),
                          shuffle = training_args['shuffle'], 
                          verbose = training_args['verbose'])
            # Note: The states are not reset at the end of the last training epoch
            model_eval = model.evaluate(X_val, Y_val)
            val_score.append({model_args['loss']: model_eval[0], 'mae': model_eval[1]})
    
    # loops > 1 and epochs=1 when stateful=True
    for loop in range(training_args['loops']):
        model.layers[0].reset_states()
        model.fit(X, 
                  Y, 
                  epochs = training_args['epochs'], 
                  batch_size = training_args['batch_size'],
                  shuffle = training_args['shuffle'], 
                  verbose = training_args['verbose'])
    # Note: The states are not reset at the end of the last training epoch
    
    return model, (X, Y), val_score

def prepare_data_for_LSTM(time_series,
                          model_args):
    '''
    Transform the time series into a 3D input for the LSTM
    
    Inputs
    ----------
    time_series: Pandas DataFrame with dates as indices
                 Time series with data the model will be fitted to.
                 It is assumed to be scaled and stationary.
            
    Outputs
    ----------
    X:           A numpy array of dimension (samples, n_timesteps, n_features)
    y:           A numpy array of dimension (samples, horizon , 1 or 2)
    '''
    
    seq_len = model_args.get('seq_len', 5)
    horizon = model_args.get('horizon', 1)
    m = len(time_series) - seq_len - horizon + 1
    n_features = model_args.get('n_features', time_series.shape[1])
    
    X = np.zeros((m, seq_len, n_features))
    Y = np.zeros((m, horizon, 1)) 
    
    for i in range(m):
        X[i, :, :] = time_series.iloc[i:i+seq_len, :n_features].values
        Y[i, :, 0] = time_series['y'].iloc[i+seq_len:i+seq_len+horizon].values
    
    return X, Y

def configure_training_args(training_args):
    training_args = defaultdict(int, training_args)
    shuffle = bool(training_args['shuffle']); training_args['shuffle'] = shuffle
    epochs = training_args['epochs']
    if epochs == 0:
        epochs = 100; training_args['epochs'] = epochs
    batch_size = training_args['batch_size']
    if batch_size == 0:
        batch_size = 32; training_args['batch_size'] = batch_size
    loops = 1
    if bool(training_args['stateful_training']):
        batch_size = 1
        shuffle = False
        loops = training_args['epochs']
        epochs = 1
    training_args['loops'] = loops
    return training_args


# Testing prepare_data_for_LSTM
if (__name__ == '__main__'):
    from src.data.debug_dataset import generate_debug_time_series
    debug_series = generate_debug_time_series()
    debug_data = prepare_data_for_LSTM(debug_series, 
                                       {'seq_len': 30,
                                        'horizon': 14})

# Testing fit_LSTM and predict_LSTM
if (__name__ == '__main__'):
    
    
    ## 1) Preparing the arguments for fit_LSTM
    from src.data.debug_dataset import generate_debug_time_series
    from src.data.prepare_dataset import process_data
    unscaled_debug_series = generate_debug_time_series()
    debug_series, scaler, first_values = process_data(unscaled_debug_series,
                                                      0,
                                                      direction='pack')
    model_args = defaultdict(int)
    arguments = {'seq_len': 30,
                 'n_features': 1,
                 'n_a': 16,
                 'learning_rate': 0.0002,
                 'loss': 'mse',
                 'stateful_training': False,
                 'stateful_inference': False,
                 'horizon': 120,
                 'autofit': False}
    for key, value in arguments.items():
        model_args[key] = value
    training_args = defaultdict(int)
    arguments = {'epochs': 100,
                 'verbose': 0}
    for key, value in arguments.items():
        training_args[key] = value
    # Arguments ready
    
    
    ## 2) Testing fit_LSTM
    debug_model, debug_training_data, val_score = fit_LSTM(debug_series,
                                                           model_args,
                                                           training_args)
    debug_model.save('debug_files/debug_LSTM_model.keras')
    # (Testing the effect of loading the model)
    # from tensorflow.keras.model import load_model
    # model = load_model('debug_files/debug_LSTM_model.keras')
    
    # (Testing prepare_initial_input on real data)
    from LSTM_predict import prepare_input
    debug_initial_input = prepare_input(debug_series,
                                        model_args)
    debug_training_input = debug_training_data[0][-1].reshape(debug_initial_input.shape)
    
    # Testing the quality of the fitted model
    prediction_on_training_data = debug_model.predict_on_batch([debug_training_input])[0]
    expected_result = debug_training_data[1][-1]
    prediction_on_test_data = debug_model.predict_on_batch([debug_initial_input])[0]
    
    
    ### 3) Testing LSTM_predict
    current_date = debug_series.index[-1]
    horizon = model_args['horizon']
    from LSTM_predict import predict_LSTM, predictor_model, form_dataframe_from_prediction
    detailed_debugging = False
    if detailed_debugging:
        # Testing predictor_model
        predictor = predictor_model(debug_model,
                                    model_args,
                                    horizon)
        prediction_on_training_data = predictor.predict_on_batch([debug_training_input])
        prediction_on_testing_data = predictor.predict_on_batch([debug_initial_input])
        
        # Testing form_dataframe_from_prediction
        forecast_on_training_data = form_dataframe_from_prediction(prediction_on_training_data, 
                                                                   model_args, 
                                                                   current_date, 
                                                                   len(debug_training_data[0]))
        forecast_on_testing_data = form_dataframe_from_prediction(prediction_on_testing_data, 
                                                                  model_args, 
                                                                  current_date, 
                                                                  len(debug_training_data[0]))
        # Unpacked forecasts
        unscaled_forecast_on_training_data = process_data(forecast_on_training_data,
                                                          0,
                                                          'unpack',
                                                          scaler,
                                                          first_values)
        unscaled_forecast_on_testing_data = process_data(forecast_on_testing_data,
                                                         0,
                                                         'unpack',
                                                         scaler,
                                                         first_values)
    
    # Testing predict_LSTM
    print(f'\nNow forecasting the {horizon} next days')
    forecast = predict_LSTM(debug_series, 
                            debug_model, 
                            model_args, 
                            current_date)
    
    unscaled_forecast = process_data(forecast, 0, 'unpack', scaler, first_values)
    
    # Visualizing the results
    from matplotlib import pyplot as plt
    plot_start_date = pd.to_datetime('2023-01-01')
    
    plt.figure(figsize=(10, 6))
    plt.plot(unscaled_debug_series[unscaled_debug_series.index>=plot_start_date], label='History', color='blue')
    plt.plot(unscaled_forecast[unscaled_forecast.index>=plot_start_date], label='Forecast', color='red')
    
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.title('Forecasting a cosinus', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    
    plt.show()
            