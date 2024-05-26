import pandas as pd
import numpy as np
from collections import defaultdict
from src.models.LSTM_files.LSTM_model import LSTM_model

def fit_LSTM(time_series,
             model_args,
             training_args):
    '''
    Fits the forecaster's model.
    
    Inputs
    ----------
    time_series: Pandas DataFrame with dates as indices
        Time series with with engineered features. Training set
                 
    model_args: defaultdict(int) with strings as keys
        Model arguments.
    
    training_args: defaultdict(int) with strings as keys
        Training arguments.
            
    Outputs
    ----------
    model: Keras model
        The fitted model
    
    (X, Y): Tuple of numpy arrays
        Training data for the LSTM
    
    train_score: List
        Training score
    '''
    
    ## 1) Data Preparation for LSTM
    X, y = prepare_data_for_LSTM(time_series,
                                 model_args)
    
    ## 2) Create the model in training mode
    model = LSTM_model(model_args, training = True)
    
    ## 3) Configure the training arguments
    training_args = configure_training_args(training_args)
    
    ## 4) Fit the model to the data
    
    # loops > 1 and epochs=1 when stateful=True
    for loop in range(training_args['loops']):
        model.fit(X, 
                  y, 
                  epochs = training_args['epochs'], 
                  batch_size = training_args['batch_size'],
                  shuffle = training_args['shuffle'], 
                  verbose = training_args['verbose'])
        model.layers[0].reset_states()
    
    train_score = model.evaluate(X, y)
    # To set the cell states
    
    return model, (X, y), train_score

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
    Y:           A numpy array of dimension (samples, horizon , 1)
    '''
    
    seq_len = model_args.get('seq_len', 5)
    m = len(time_series) - seq_len
    n_features = time_series.shape[1]-1
    
    X = np.zeros((m, seq_len, n_features))
    y = np.zeros((m,)) 
    
    for i in range(m):
        X[i, :, :] = time_series.drop(columns=['Return']).iloc[i:i+seq_len].values
        y[i] = time_series['Return'].iloc[i+seq_len]
    
    return X, y

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
                                       {'seq_len': 30})

# Testing fit_LSTM and predict_LSTM
if (__name__ == '__main__'):
    
    
    ## 1) Preparing the arguments for fit_LSTM
    from src.data.debug_dataset import generate_debug_time_series
    debug_series = generate_debug_time_series()
    model_args = defaultdict(int)
    arguments = {'seq_len': 30,
                 'horizon': 1,
                 'n_a': 16,
                 'learning_rate': 0.0002,
                 'loss': 'mse',
                 'stateful_training': False,
                 'stateful_inference': False,
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
    from LSTM_predict import predict_LSTM, predictor_model, form_dataframe_from_prediction
    detailed_debugging = False
    if detailed_debugging:
        # Testing predictor_model
        predictor = predictor_model(debug_model,
                                    model_args)
        prediction_on_training_data = predictor.predict_on_batch([debug_training_input])
        prediction_on_testing_data = predictor.predict_on_batch([debug_initial_input])
        
        # Testing form_dataframe_from_prediction
        forecast_on_training_data = form_dataframe_from_prediction(prediction_on_training_data, 
                                                                   model_args['horizon'], 
                                                                   current_date)
        forecast_on_testing_data = form_dataframe_from_prediction(prediction_on_testing_data, 
                                                                  model_args['horizon'], 
                                                                  current_date)
        # Unpacked forecasts
    
    # Testing predict_LSTM
    print('\nNow forecasting...')
    forecast = predict_LSTM(debug_series, 
                            debug_model, 
                            model_args, 
                            current_date,
                            1)
    
    
    # Visualizing the results
    from matplotlib import pyplot as plt
    plot_start_date = pd.to_datetime('2024-03-01')
    
    plt.figure(figsize=(10, 6))
    plt.plot(debug_series['Return'][debug_series.index>=plot_start_date], label='Return', color='blue')
    forecast_to_plot = pd.DataFrame([debug_series['Return'].iloc[-1] , forecast['Return'].iloc[0]],
                                    columns = ['Return'],
                                    index = [debug_series.index[-1], forecast.index[0]])
    plt.plot(forecast_to_plot, label='Return forecast', color='red')
    
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.title('Forecasting a cosinus', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    
    plt.show()
            