import pandas as pd
import matplotlib.pyplot as plt

class recommender():
    ''' 
    Simulates trading of stocks based on the predictions of a statistical tool.
    
    Parameters
    ----------
    oracle: An object of the class forecaster
        Responsible for stock price forecasting
                  
    trading_args: dictionary with strings as keys
        Trading arguments. Must include:
        initial_stock: Float 
            Initial stock owned in the simulation
        max_trade: Float
            Maximum number of stock units to trade at once
        intensity: Integer. 
             Parameter proportional to the quantity of stock traded per period
        min_delta: Float 
            Minimum predicted relative variation of the stock price 
            to recommend a BUY or SELL action.
    '''
    def __init__(self,
                 oracle,
                 trading_args):
        
        self.oracle = oracle
        self.trading_args = trading_args
        
    def __call__(self,
                 data_args,
                 temporal_args,
                 training_args,
                 verbose = True):
        ''' 
        Performs a series of recommendations with the following procedure loop:
        
        1) Fits predictor to the data up to current_date from this iteration 
        2) Makes a prediction of the future stock price (after current_date)
            within a specified horizon timeframe
        3) Makes a recommendation (BUY/SELL/HOLD) based on that prediction
        4) Updates current_date to the last date the prediction covered
    
        This sequence is looped up until a final end_date is reached.
            
        Inputs
        ----------
        data_args: dictionary with strings as keys
            Data arguments. Must include:
            features: pandas DataFrame
                Contains the engineered features
            close: pandas DataFrame
                Historical stock price (Close)
            t_scaler: scikit-learn scaler
                Scaler used for the return target
                
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
        
        verbose: Boolean
            Whether to print current stock price and prediction during simulation
        
        Outputs
        ----------
        A plot of the predictions compared to the real data, the financial
        balance resulting from applying the successive recommendations
        '''
        
        ## 1) Initialization
        
        # Get the temporal arguments
        start_date = temporal_args['start_date']
        end_date = temporal_args['end_date']
        
        # Get the historical data
        history = data_args['close']
        features = data_args['features']
        symbol = data_args['symbol']
        
        plot_start_date = temporal_args.get('plot_start_date', history.index[20])
        
        # Get the initial stock and its value
        initial_stock = self.trading_args['initial_stock']
        initial_price = self.get_price_from_date(history,
                                                 start_date)
        initial_value = initial_price*initial_stock
        self.trading_args['initial_value'] = initial_value
        
        # Initialize the attributes
        self.initialize_attributes(initial_value)
        self.failed_forecast = False
        
        # Initialize the current date
        current_date = start_date
        
        # Initialize a new matplotlib figure
        plt.figure()
        
        # Draw a blue vertical line that indicates {start_date}
        plt.axvline(start_date, color = 'b') 
        
        
        ## 2) Run the loop up until current_date reaches end_date
        # Each iteration is a forecast up to next_date {horizon} days ahead
        # Then current_date is updated to next_date
        
        while current_date < end_date:
            if verbose:
                print(f'\n{current_date = }')
            
            ## a) Forecasts the future for the {horizon} next days
            data_args['features'] = features[features.index<=current_date]
            forecast = self.forecast(data_args,
                                     temporal_args,
                                     training_args,
                                     current_date)
            
            # Predicted stock price {horizon} days after {current_date}
            prediction = forecast['Close'].iloc[-1]
            
            # {next_date} is the date of the last prediction
            next_date = forecast.index[-1]
            if verbose:
                print(f'\n{next_date = }')
            
            ## b) Draw a green vertical line indicating when a prediction was made
            plt.axvline(next_date, color = 'k')
            
            ## c) Update the list of predictions
            if self.predictions.empty:
                self.predictions = forecast
            else:
                self.predictions = pd.concat([self.predictions, forecast], 
                                             ignore_index = False)
            
            # If forecast has failed
            if forecast.isna().values.any():
                print('\nError: Forecast failed')
                self.failed_forecast = True
                break
            
            ## d) Perform a recommendation based on the stock price prediction
            self.recommend(history, prediction, current_date, next_date, verbose)
            
            ## e) Update current_date for the next iteration
            current_date = next_date
            
        current_price = self.get_price_from_date(history,
                                                 current_date)
        
        initial_price = self.get_price_from_date(history,
                                                 start_date)
        
        forecast, real_values_adjusted = self.adjust_dates(self.predictions,
                                                           history,
                                                           start_date,
                                                           end_date)
        
        baseline_forecast = self.baseline(real_values_adjusted,
                                          initial_price)
        
        self.MAE, self.mae = self.compute_MAE(real_values_adjusted, 
                                              forecast)
        
        self.baseline_MAE, self.baseline_mae = self.compute_MAE(real_values_adjusted, 
                                                                baseline_forecast)
        
        self.MSE, self.mse = self.compute_MSE(real_values_adjusted, 
                                              forecast)
        
        self.baseline_MSE, self.baseline_mse = self.compute_MSE(real_values_adjusted, 
                                                                baseline_forecast)
        
        if self.failed_forecast:
            return None
        
        ## 3) Print information to help evaluate the trading recommendations
        
        if verbose:
            self.print_trading_performance(current_price,
                                           temporal_args)
        
        
        ## 4) Plot the predictions compared to real data
        
        # Plot the Bollinger Bands in blue (bottom) and red (top)
        rolling = history[history.index>=plot_start_date].rolling(20)
        mean, std = rolling.mean(), rolling.std()
        bolling_min = (mean-2*std).squeeze()
        bolling_max = (mean+2*std).squeeze()
        plt.plot(bolling_min, color = 'b', label = 'Top of Bollinger Band')
        plt.plot(bolling_max, color = 'r', label = 'Bottom of Bollinger Band')
        
        # Plot the predictions (full green) VS the real data (dotted black)
        plot_pred = self.predictions[self.predictions.index>=plot_start_date]
        plot_hist = history[history.index>=plot_start_date][['Close']]
        plt.plot(plot_pred, color = 'g', label = 'Predictions')
        plt.plot(plot_hist, ':', color = 'k', label = 'History')
        
        # Legend, axis labels and figure title
        plt.legend()
        plt.xticks(fontsize = 8) 
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.title(f'Forecast and real data compared for {symbol}')
        
        # Show
        plt.show()
        
        return None
    
    def initialize_attributes(self,
                              initial_value):
        '''
        Initializes the recommender's attributes.
        These attributes are metrics used to simulate trading actions
        and evaluate the future recommendations made by the recommender.
        
        Inputs
        ----------
        initial_value: Float
            Value of Initially Owned Stock
        
        Attributes initialized
        ----------
        metrics: Dictionary with strings as keys
            Contains the following:
            value: Float
                Value of Currently Owned Stock
            stock: Float
                Number of units of Currently Owned Stock
            gain: Float
                Gains made from last trading action
                Positive if SELL
                Negative if BUY
            net_gain: Float
                Gains made from last trading action subtracted by current value of stock traded
                Positive if BUY and stock value increased or SELL and stock value decreased
                Negative if BUY but stock value decreased or SELL but stock value increased
        
        metrics_history: Dictionary with strings as keys
            Contains the following:
            value_history: List
                Evolution of the value of Currently Owned Stock
            gain_history: List
                List of subsequent gains made from each trading action
            net_gain_history: List
                List of subsequent net gains made from each trading action
            action_history: List
                List of subsequent trading actions performed
            
        predictions: pandas DataFrame
            DataFrame with stock price predictions used for recommendation
        '''
        
        self.metrics = {
            'value': initial_value,
            'stock': self.trading_args['initial_stock'],
            'gain': 0,
            'net_gain': 0
            }
        
        self.metrics_history = {
            'value_history': [initial_value],
            'gain_history': [0],
            'net_gain_history': [0],
            'value_change_history': [0],
            'action_history': ['HOLD']
            }
        
        self.predictions = pd.DataFrame()
        
        return None
    
    def get_price_from_date(self,
                            history,
                            date):
        '''
        Returns the historical price at the nearest date to the one requested 
        with a known stock price.
        
        Inputs
        ----------
        features: Pandas DataFrame
            Time series engineered features related to the stock forecasted
        date: Timestamp
            Date requested
        
        Outputs
        ----------
        price: Float
            Stock price at the nearest date to {date} with a known stock price.
        '''
        date = pd.Series((history.index-date).days).abs().idxmin()
        price = history['Close'].iloc[date]
        return price
        
    def recommend(self, 
                  history, 
                  predicted_next_price,
                  current_date, 
                  next_date,
                  verbose):
        '''
        Performs a trading recommendation based on the predictor's forecast.
        Updates the recommender attributes related to stock held, 
        profit made, and history of recommendations performed.
        
        Arguments
        ----------
        history: Pandas DataFrame
            Time series of historical Close stock prices
        predicted_next_price: Float
            Most recent (latest) predicted stock price
        current_date: Timestamp
            Date up to which stock price data was known
        next_date: Timestamp
            Date up to which the stock price is predicted
        verbose: Boolean
            Whether to print current stock price and prediction
        
        Updates
        ----------
        self.stock: Float
            Simulated stock quantity
        self.value: Float
            Simulated stock value
        self.gains: Float
            Simulated gains from stock bought or sold
        self.net: Float
            Simulated net gains from stock bought or sold (compared to no trading)
        self.action_history: List
            List of simulated trading actions
        self.value_history: List
            List of simulated self.value values
        self.gain_history: List
            List of simulated self.gain values
        self.net_gain_history: List
            List of simulated self.net values
        '''
        
        ## 1) Retrieve relevant information
        # Recover trading arguments
        max_trade = self.trading_args['max_trade']
        min_delta = self.trading_args['min_delta']
        intensity = self.trading_args['intensity']*max_trade
        
        # Get current stock price
        current_price = self.get_price_from_date(history,
                                                 current_date)
        
        # Get future stock price
        actual_next_price = self.get_price_from_date(history,
                                                     next_date)
        
        if verbose:
            print(f'\n{current_price = }')
            print(f'{predicted_next_price = }')
            print(f'{actual_next_price = }')
        
        # Compute predicted relative price change to help with decision making
        if current_price == 0:
            relative = 1
        else:
            relative = (predicted_next_price-current_price)/current_price 
        
        ## 2) Select an action based on the predicted relative price change
        action = ["SELL", "BUY"]
        if abs(relative) >= min_delta: # If price change exceeds the threshold
            if current_price == 0:
                t = 1000
            else:
                t = predicted_next_price/current_price 
            sign = int(t>1/t) - int(t<1/t) # Equals 1 if predicted_next_price>current_price, else -1
            trade = max(-self.metrics['stock'], sign*min(intensity*(max(t, 1/t)-1), max_trade))
            # The absolute amount of stock traded (= intensity*(max(t, 1/t)-1)) 
            # is higher with larger relative price change (= max(t, 1/t)-1)
            self.metrics_history['action_history'].append(action[int(relative>0)]) # BUY or SELL
        else: # If predicted relative price change is below the threshold
            trade = 0
            self.metrics_history['action_history'].append("HOLD")
        
        ## 3) Apply the action
        value_precision = 1
        # Update trading metric values
        self.metrics['stock'] += trade
        self.metrics_history['value_change_history'].append(round(self.metrics['stock']*actual_next_price-self.metrics['value'], value_precision))
        self.metrics['value'] = self.metrics['stock']*actual_next_price
        self.metrics['gains'] = -trade*current_price
        self.metrics['net_gains'] = trade*(actual_next_price-current_price)
        # Update historical trading metric values
        self.metrics_history['value_history'].append(round(self.metrics['value'], value_precision))
        self.metrics_history['gain_history'].append(round(self.metrics['gains'], value_precision))
        self.metrics_history['net_gain_history'].append(round(self.metrics['net_gains'], value_precision))
        
        return None
    
    def forecast(self, 
                 data_args,
                 temporal_args,
                 training_args,
                 current_date):
        '''
        Performs a trading recommendation based on the forecaster's forecast.
        Updates the recommender attributes related to stock held, 
        profit made, and history of recommendations performed.
        
        Inputs
        ----------
        data_args: dictionary with strings as keys
            Data arguments. Must include:
            order: positive integer
                Order of differencing required to make the time series stationary
                
        training_args: dictionary with strings as keys
            Training arguments. May include:
            epochs:
                Number of epochs when a keras model fits to the data
        
        current_date: Timestamp
            Date up to which the stock price data is learned by the model
        
        Outputs
        ----------
        forecast: Pandas DataFrame
            Dataframe with forecast of stock price (Close)
        '''
        # Defining the temporal arguments to call the oracle
        temporal_args['current_date'] =  current_date
        
        # Calling the oracle to forecast the return
        forecast = self.oracle(data_args,
                               temporal_args,
                               training_args)
        
        # Computes the forecasted Close price
        current_close  = self.get_price_from_date(data_args['close'], current_date)
        predicted_close = (forecast['Return'].iloc[-1] + 1)*current_close
        
        # Convert to DataFrame
        forecast = pd.DataFrame(predicted_close, 
                                columns = ['Close'], 
                                index = forecast.index)
        
        return forecast
    
    def print_trading_performance(self,
                                  current_price,
                                  temporal_args):
        '''
        Prints a number of useful trading performance metrics to the console.
        This function is called after a trading simulation is completed.
        
        Inputs
        ----------
        current_price: Float
            The stock price at the end_date the trading simulation terminated
            
        temporal_args: dictionary with strings as keys
            Temporal arguments.
        '''
        
        ## Retrieving information
        start_date = temporal_args['start_date']
        end_date = temporal_args['end_date']
        initial_value = self.trading_args['initial_value']
        initial_stock = self.trading_args['initial_stock']
        stock = self.metrics['stock']
        value = self.metrics['value']
        gains = self.metrics['gains']
        net_gains = self.metrics['net_gains']
        action_history = self.metrics_history['action_history'][1:]
        value_history = self.metrics_history['value_history'][1:]
        gain_history = self.metrics_history['gain_history'][1:]
        net_gain_history = self.metrics_history['net_gain_history'][1:]
        value_change_history = self.metrics_history['value_change_history'][1:]
        total_change_history = [a+b for a,b in zip(gain_history, value_change_history)]
        total_value_history = [a+b for a,b in zip(gain_history, value_history)]
        # Trading performance
        stock_precision = 3
        value_precision = 1
        print(f'\nThe Wallet gains per trade is (positive values = SELL)\n{list(zip(action_history, gain_history))}')
        print(f'\nThe Stock Value evolution after each trade is\n{list(zip(action_history, value_history))}')
        print(f'\nStock + Wallet value variation after trading: \n{list(zip(action_history, total_change_history))}')
        print(f'\nStock + Wallet Net Gains related to stock traded: \n{list(zip(action_history, net_gain_history))}')
        print(f'with sum of {sum(net_gain_history):.{value_precision}f}')
        print(f'\nInitial stock (Quantity = {initial_stock}) value on {start_date}: {initial_value:.{value_precision}f}')
        print(f'\nInitial stock (Quantity = {initial_stock}) value on {end_date}: {(initial_stock*current_price):.{value_precision}f}')
        print(f'\nFinal stock (Quantity = {stock:.{stock_precision}f}) value on {end_date}: {value:.{value_precision}f}')
        print(f'\nExpenses from trading: {-sum(gain_history):.{value_precision}f}')
        print(f'\nBalance compared to initial stock value on {end_date} (value owned if no trading) = {(sum(gain_history) + value - initial_stock*current_price):.{value_precision}f}')
        print(f'\nBalance compared to initial stock value on {start_date}: {(sum(gain_history) + value - initial_value):.{value_precision}f}')
        # Forecasting performance
        print(f'\n{self.mse = }')
        print(f'\n{self.mae = }')
        print(f'\n{self.MSE = }') 
        print(f'\n{self.MAE = }')
        print(f'\n{self.baseline_MSE = }')
        print(f'\n{self.baseline_MAE = }')
    
    def adjust_dates(self, 
                     forecast, 
                     history,
                     start_date,
                     end_date):
        """
        Align the dates of a history dataframe to those of the forecast one
        using linear interpolation for missing dates.
    
        Inputs
        ----------
        forecast: DataFrame 
            Forecast of stock prices (Close)
        history: DataFrame
            History of stock prices (Close)
        start_date: Timestamp
            First date from which the predictions were requested to start
        end_date: Timestamp
            Last date up to which the predictions were requested to extend to
    
        Outputs
        ----------
        forecast_adjusted: DataFrame 
            The adjusted DataFrame
        """
        end_date = min(end_date, history.index[-1], forecast.index[-1])
        hmask = (history.index >= start_date) & (history.index <= end_date) & (history.index >= forecast.index[0])
        fmask = (forecast.index >= start_date) & (forecast.index <= end_date)
        
        real_values = history[hmask]
        forecast = forecast[fmask]
    
        real_values_adjusted = real_values.reindex(forecast.index).interpolate(method='time')
    
        return forecast, real_values_adjusted
    
    def baseline(self,
                 real_values,
                 first_price):
        '''
        Computes a baseline forecast based on the real stock price values.
        The baseline is just a constant function.
        
        Inputs
        ----------
        real_values: Pandas DataFrame
            The stock price historical data
        
        Outputs
        ----------
        baseline_forecast: Pandas DataFrame
            The baseline forecast
        '''
        forecasted_values = [first_price] + list(real_values['Close'].values[:-1])
        baseline_forecast = pd.DataFrame(forecasted_values,
                                         index = real_values.index,
                                         columns = ['Close'])
        return baseline_forecast

    def compute_MAE(self, 
                    history, 
                    forecast):
        '''
        Returns the forecast Mean Absolute Error between start_date and end_date
        
        Inputs
        ----------
        history: Pandas DataFrame
            History of stock prices (Close)
        forecast: Pandas DataFrame
            Dataframe with predictions of stock price and other relevant data
            
        Outputs
        ----------
        MAE: Tuple
            The first element is the rounded overall Mean Average Error
            The second element is the Mean Average Error at every time step
        '''
        error = history.subtract(forecast).abs()
        MAE = error.mean(skipna=not(self.failed_forecast)).iloc[0], error
        return MAE

    def compute_MSE(self, 
                    history, 
                    forecast):
        '''
        Returns the forecast Mean Square Error between start_date and end_date
        
        Inputs
        ----------
        history: Pandas DataFrame
            History of stock prices (Close)
        forecast: Pandas DataFrame
            Dataframe with predictions of stock price and other relevant data
            
        Outputs
        ----------
        MAE: Tuple
            The first element is the rounded overall Mean Average Error
            The second element is the Mean Average Error at every time step
        '''
        mse = history.subtract(forecast).abs()**2
        MSE = mse.mean(skipna=not(self.failed_forecast)).iloc[0], mse
        return MSE