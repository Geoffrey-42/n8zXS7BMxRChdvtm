from collections import defaultdict
import pandas as pd

class forecaster():
    '''
    Forecasts stock prices based on historical data.
    
    Parameters
    ----------
    model_name: String
        Name of the model chosen for forecasting
    model_args: dictionary
        Contains arguments for the model definition
    '''
    def __init__(self,
                 model_name,
                 model_args = {}):
        if model_name.lower() == 'lstm':
            self.model_name = 'lstm'
            from src.models.LSTM_files.LSTM_setup import LSTM_config
            model_args = LSTM_config(defaultdict(int, model_args))
            self.model_args = model_args
        elif model_name.lower() == 'fbprophet':
            self.model_name = 'fbprophet'
            self.model_args = model_args
        elif model_name.lower() == 'autoreg':
            self.model_name = 'autoreg'
            self.model_args = model_args
        elif model_name.lower() == 'arima':
            self.model_name = 'arima'
            self.model_args = model_args
        elif model_name.lower() == 'sarimax':
            self.model_name = 'sarimax'
            self.model_args = model_args
        elif model_name.lower() == 'simpleexpsmoothing':
            self.model_name = 'simpleexpsmoothing'
            self.model_args = model_args

    def __call__(self,
                 data_args,
                 temporal_args,
                 training_args = {}):
        '''
        Generates a forecast for {horizon} days into the future.
        
        Inputs
        ----------
        data_args: dictionary with strings as keys
            Data arguments. Must include:
            features: pandas DataFrame
                Time series engineered features related to the stock forecasted
                
        temporal_args: dictionary with strings as keys
            Temporal arguments. Must include:
            start_date: pandas Timestamp
                First date up to which the predictor will be fitted
            end_date: pandas Timestamp
                Last date that the predictions extend to
            horizon: positive integer
                Indicates up to how many days the forecast will extend to
        
        training_args: dictionary with strings as keys
            Training arguments.
                
        Outputs
        ----------
        forecast: Pandas DataFrame
            DataFrame with the forecast of stock prices and other relevant data
        '''
        
        training_args = defaultdict(int, training_args)
        
        scaled_forecast = self.do_forecast(data_args,
                                           temporal_args,
                                           training_args)
        
        # Unscale the result
        t_scaler = data_args['t_scaler']
        forecast = pd.DataFrame(t_scaler.inverse_transform(scaled_forecast),
                                columns = ['Return'],
                                index = scaled_forecast.index)
        
        return forecast
    
    def do_forecast(self,
                    data_args,
                    temporal_args,
                    training_args):
        '''
        Fits the forecaster's model to the data up to {current_date}.
        Then, produces a forecast with a {horizon}-day horizon.
        
        Inputs
        ----------
        Same as __call__
        
        Outputs
        ----------
        Same as __call__
        '''
        
        args = {
            'data_args': data_args,
            'temporal_args': temporal_args,
            'model_args': self.model_args,
            'training_args': training_args
            }
        
        if self.model_name == 'lstm':
            from src.models.LSTM import LSTM_forecast
            return LSTM_forecast(args)
        elif self.model_name == 'fbprophet':
            from src.models.fbprophet import fbprophet_forecast
            return fbprophet_forecast(args)
        elif self.model_name == 'autoreg':
            from src.models.autoreg import autoreg_forecast
            return autoreg_forecast(args)
        elif self.model_name == 'arima':
            from src.models.arima import arima_forecast
            return arima_forecast(args)
        elif self.model_name == 'sarimax':
            from src.models.sarimax import sarimax_forecast
            return sarimax_forecast(args)
        elif self.model_name == 'simpleexpsmoothing':
            from src.models.simpleexpsmoothing import SimpleExpSmoothing_forecast
            return SimpleExpSmoothing_forecast(args)
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        