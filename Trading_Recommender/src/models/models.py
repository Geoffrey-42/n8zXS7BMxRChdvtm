runfile('src/data/make_dataset.py')

class forecaster():
    '''
    Forecasts stock prices based on historical data.
    
    Parameters
    ----------
    model_name: String, model chosen for forecasting
    args: Defaultdict(int), contains arguments for the model definition
    '''
    def __init__(self, model_name, args = defaultdict(int)):
        if model_name.lower() == 'lstm':
            self.model_name = 'lstm'
            
            # Define the LSTM layer
            n_a = args['n_a']
            if n_a == 0:
                n_a = 32; args['n_a'] = n_a
            stateful = bool(args['stateful']); args['stateful'] = stateful
            LSTM_layer = LSTM(n_a, 
                              activation = 'relu', 
                              return_state = True, 
                              return_sequences = False,
                              stateful = stateful, 
                              name = 'LSTM_layer')
            # The stateful mode can be activated to find dependencies between input sequences:
            # The last state for each sample at index i in a batch will be used 
            # as initial state for the sample of index i in the following batch.
            # In that case, batch_size needs to be 1.
            
            # Define the Dense layer
            include_dates = bool(args['include_dates']); args['include_dates'] = include_dates
            n_features = min(args['n_features'], 1+int(include_dates)); args['n_features'] = n_features
            dense = Dense(n_features, 
                          activation = 'relu', 
                          name = 'Dense_layer')
            
            # Create the model
            batch_size = 32 - 31*int(stateful) # 32 or 1 if stateful mode on
            self.batch_size = batch_size
            seq_len = args['seq_len']
            if seq_len == 0:
                print("Warning: argument 'seq_len' was not specified in the args defaultdict\n")
                print("The sequence length of the training examples fed into the LSTM will be set to 60 by default\n")
                seq_len = 60; args['seq_len'] = seq_len
            self.model = self.LSTM_model(seq_len,
                                         n_features,
                                         LSTM_layer, 
                                         dense)
            
            # Define the various custom losses that may be selected
            def huber_loss(y_true, y_pred, delta=1.0):
                error = y_true - y_pred
                quadratic_term = tf.minimum(tf.abs(error), delta)
                linear_term = tf.abs(error) - quadratic_term
                return 0.5 * tf.square(quadratic_term) + delta * linear_term

            def quantile_loss(q, y_true, y_pred):
                error = y_true - y_pred
                return K.mean(K.maximum(q * error, (q - 1) * error), axis=-1)
            
            def logcosh_loss(y_true, y_pred):
                log_pred = K.log(y_pred + K.epsilon())
                log_true = K.log(y_true + K.epsilon())
                loss = K.mean(K.log(K.cosh(log_pred - log_true) + K.epsilon()), axis=-1)
                return loss

            def log_loss(y_true, y_pred):
                error = y_true - y_pred
                return K.mean(K.log(1 + K.abs(error)), axis=-1)
            
            # Select the loss
            loss = args['loss']
            if loss in {'huber', 'quantile', 'logcosh', 'log'}:
                loss = loss + '_loss'
            loss_set = {'huber_loss', 'quantile_loss', 'logcosh_loss', 'log_loss',
                        'mean_squared_error', 'mse', 'mean_absolute_error', 'mae', 
                        'mean_absolute_percentage_error', 'mape'
                       }
            if loss not in loss_set:
                print('The loss was either not specified or not supported by this recommender.\n')
                print(f'Supported losses are {loss_set}\n')
                print('Loss will be set by default to the huber loss\n')
                loss = 'huber_loss'
            args['loss'] = loss
            
            # Compile the model
            learning_rate = args['learning_rate']
            if learning_rate == 0:
                learning_rate = 0.01; args['learning_rate'] = learning_rate
            decay = args['decay']
            if decay == 0:
                decay = 0.99; args['decay'] = decay
            self.model.compile(loss = loss, #huber loss, log cost, quantile loss fn
                               optimizer = Adam(learning_rate = learning_rate,
                                                beta_1=0.9, 
                                                beta_2=0.999,
                                                decay=decay, 
                                                amsgrad=False),
                               metrics = ['mean_absolute_error'])
            self.args = args
            # print(self.model.summary())
        elif model_name.lower() == 'prophet':
            self.model_name = 'prophet'
        elif model_name.lower() == 'autoreg':
            self.model_name = 'autoreg'
        elif model_name.lower() == 'arima':
            self.model_name = 'arima'
        elif model_name.lower() == 'sarimax':
            self.model_name = 'sarimax'
        elif model_name.lower() == 'simpleexpsmoothing':
            self.model_name = 'simpleexpsmoothing'

    def __call__(self, 
                 history, current_date, 
                 periods, interval):
        '''
        Fits the forecaster's model to the data up to current_date.
        Then, performs a forecast every {interval} days, {periods} times.
        
        Arguments
        ----------
            history: Pandas series, time series the model will be fitted to.
            current_date: Until that date, the model will be fitted to the data.
            interval: Integer, number of days between each prediction
            periods: Integer, number of predictions
        
        Returns
        ----------
            forecast: A pandas series with forecast dates as indexes and corresponding forecast values.
        '''
        # Extract the data to fit the model to
        to_fit = history[history.index<=current_date]
        if self.model_name == 'lstm':
            LSTM_layer, dense, scaler, inputs = self.fit_LSTM(to_fit)
            
            forecast = self.predict_LSTM(inputs,
                                         scaler,
                                         LSTM_layer,
                                         dense,
                                         current_date,
                                         interval,
                                         periods)

            prediction = forecast.iloc[-1] # Predicted future stock price
            next_date = forecast.index[-1]
        elif self.model_name == 'prophet':
            predictor = Prophet()
            predictor.fit(to_fit)
            # Predictor is now fitted to data prior to current_date
            freq_dict = {1:"d", 7:"w", 30:"m", 90: 'Q'}
            future = predictor.make_future_dataframe(periods = periods, 
                                                     freq = freq_dict[interval], 
                                                     include_history = False)
            forecast = predictor.predict(future)
            print(forecast)
            forecast.rename(columns={'yhat': 'y'}, inplace = True)
            prediction = forecast['y'].iloc[-1] # Predicted future stock price
            next_date = forecast['ds'].iloc[-1]
            print(f'\ncurrent_date = {current_date}', f'\nfuture={future}', f'\nforecast={forecast}')
            print(f"last forecast = {forecast.iloc[-1]['ds']}")
        elif self.model_name == 'autoreg':
            mask = history.index <= current_date
            to_fit = history[mask]
            to_fit.index = pd.DatetimeIndex(to_fit.index).to_period('D')
            result = adfuller(to_fit.diff().dropna())
            print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % result[1])
            # sel = ar_select_order(to_fit, 13, glob = True, seasonal = False)
            # print(sel.ar_lags)
            # predictor = sel.model.fit()
            predictor = AutoReg(to_fit, lags = [1])
            params = predictor.fit().params
            # Predictor is now fitted to data prior to current_date
            start = history.loc[(history['ds']-current_date).abs().idxmin()]['ds']
            next_date = current_date+pd.Timedelta(periods*interval, 'D')
            end = history.loc[(history['ds']-next_date).abs().idxmin()]['ds']
            forecast = predictor.predict(params, 
                                         start = start, 
                                         end = end).to_timestamp()
            forecast = forecast[forecast.index>=start]
            prediction = forecast.iloc[-1] # Predicted future stock price
            next_date = forecast.index[-1]
        elif self.model_name == 'arima':
            mask = history.index <= current_date
            to_fit = history[mask]
            to_fit = to_fit.set_index('ds')
            to_fit.index = pd.DatetimeIndex(to_fit.index).to_period('D')
            predictor = ARIMA(to_fit)
            # Predictor is now fitted to data prior to current_date
            forecast = predictor.predict()
            forecast = forecast[forecast.index>=start]
            prediction = forecast.iloc[-1] # Predicted future stock price
            next_date = forecast.index[-1]
        elif self.model_name == 'sarimax':
            mask = history.index <= current_date
            to_fit = history[mask]
            to_fit = to_fit.set_index('ds')
            to_fit.index = pd.DatetimeIndex(to_fit.index).to_period('D')
            predictor = SARIMAX(to_fit)
            # Predictor is now fitted to data prior to current_date
            forecast = predictor.predict()
            forecast = forecast[forecast.index>=start]
            prediction = forecast.iloc[-1] # Predicted future stock price
            next_date = forecast.index[-1]
        elif self.model_name == 'simpleexpsmoothing':
            mask = history.index <= current_date
            to_fit = history[mask]
            to_fit = to_fit.set_index('ds')
            to_fit.index = pd.DatetimeIndex(to_fit.index).to_period('D')
            predictor = SimpleExpSmoothing(to_fit)
            # Predictor is now fitted to data prior to current_date
            forecast = predictor.predict()
            forecast = forecast[forecast.index>=start]
            prediction = forecast.iloc[-1] # Predicted future stock price
            next_date = forecast.index[-1]

        return forecast, prediction, next_date
    
    def LSTM_model(self,
                   seq_len, n_features,
                   LSTM_layer, dense):
        '''
        Fits the forecaster's model.
        
        Arguments:
            to_fit: Time series with data the model will be fitted to.
            
        Returns:
            LSTM_layer: A trained LSTM layer
            dense: A trained Dense layer
            scaler: A Scaler that has been fitted to the data
            inputs: A list containing
                x0: Array, sequence that follows the last labeled sequence.
                    Will be used as an input to predict_LSTM
                a0: Array, initial LSTM hidden state
                c0: Array, initial LSTM cell state
        '''
        n_a = LSTM_layer.units
        
        X = Input(batch_shape = (self.batch_size, seq_len, n_features))
        
        a0 = Input(batch_shape = (self.batch_size, n_a), name = 'a0')
        c0 = Input(batch_shape = (self.batch_size, n_a), name = 'c0')
        
        a = a0
        c = c0
        
        outputs_h = []
        
        # for step in range(seq_len):
            # x = X[:, step, :] # shape = (None, n_features)
            # x = Reshape([1, n_features])(x) # shape = (None, 1, n_features)
            # _, a, c = LSTM_layer(inputs = x, initial_state = [a, c])
            
        _, a, c = LSTM_layer(inputs = X, initial_state = [a, c])
        
        a = Dropout(0.1)(a)
        
        outputs = Reshape([n_features])(dense(a)) # shape = (None, n_features)
        model = Model(inputs = [X, a0, c0], outputs = outputs)
        return model
    
    def fit_LSTM(self, to_fit):
        '''
        Fits the forecaster's model.
        
        Arguments:
            to_fit: Time series with data the model will be fitted to.
            
        Returns:
            LSTM_layer: A trained LSTM layer
            dense: A trained Dense layer
            scaler: A Scaler that has been fitted to the data
            x0: Array, sequence that follows the last labeled sequence.
                Will be used as an input to predict_LSTM
            a0: Array, initial LSTM hidden state
            c0: Array, initial LSTM cell state
        '''
        args = self.args
        
        # Whether the time series will be differenced to make the time series stationary
        if bool(args['difference']): 
            to_fit = to_fit.diff().fillna(method='ffill').fillna(method='bfill')
        
        # Normalize the stock price values
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = RobustScaler()
        normalized_data = scaler.fit_transform(to_fit.values.reshape(-1, 1))
        
        # Transform the time series data into a 3D structure adapted to LSTMs
        seq_len = args['seq_len']
        m = len(normalized_data) - seq_len # Number of samples
        extra_m = m%self.batch_size
        m = m-extra_m
        # print(f'{m = }')
        to_fit = to_fit.iloc[extra_m:]
        normalized_data = normalized_data[extra_m:]
        n_features = min(args['n_features'], 1)
        include_dates = bool(args['include_dates'])
        reference_date = to_fit.index[0] # Most ancient date 
        X = np.zeros((m, seq_len, n_features)) # shape = (samples, n_timesteps, n_features)
        Y = np.zeros((m, int(include_dates)+1)) # shape = (samples, 1 or 2)
        for i in range(m):
            if include_dates:
                dates = to_fit.index[i:i+seq_len]
                for j in range(len(dates)):
                    X[i, j, 0] = (dates[j]-reference_date).days
                Y[i, 0] = to_fit.index[i+seq_len]
            sequence = normalized_data[i:i+seq_len]
            for j, data_point in enumerate(sequence):
                X[i, j, int(include_dates):] = sequence[j]
            Y[i, int(include_dates)] = normalized_data[i+seq_len][0]
            # The feature to predict is the first feature
            
        # print(f'{(X[0], Y[0]) = }')
        # print(f'{(X.shape, Y.shape) = }')
        
        # Prepare x0 for future use as an input to predict_LSTM
        x0 = X[-1:, :, :]; y = Y[-1, 0] # The last sequence will concatenate these elements
        x0 = x0[:, 1:, :] # remove left side element
        y = np.reshape([y], (1, 1, 1)) # define right side element
        x0 = np.concatenate((x0, y), axis = 1) # append right side element
        
        ## Prepare and fit the LSTM model
        n_a = args['n_a']
        a0, c0 = np.zeros((m, n_a)), np.zeros((m, n_a))
        # print(f'{(X.shape, Y.shape) = }')
        batch_size = 32; shuffle = True
        if bool(args['stateful']):
            batch_size = 1
            shuffle = False
        epochs = args['epochs']
        if epochs == 0:
            epochs = 50
        
        self.model.fit([X, a0, c0], 
                       Y, 
                       epochs = epochs, 
                       batch_size = batch_size, 
                       shuffle = shuffle, 
                       verbose = 1)
        
        if args['stateful']:
            # Do a forward pass on sequences in chronological order
            # The hidden states and cell states will be retrieved after
            self.model.predict([X, a0, c0], batch_size = 1)
            LSTM_layer = self.model.get_layer(name='LSTM_layer')
            a0, c0 = LSTM_layer.states
            # print(f'{(a0, c0) = }')
        
        LSTM_layer = self.model.get_layer(name='LSTM_layer') # Trained LSTM layer
        dense = self.model.get_layer(name='Dense_layer') # Trained dense layer
        
        inputs = [x0[0:1], a0[-1:], c0[-1:]]
        # print(f'x0 unscaled = {scaler.inverse_transform(x0[0])}')
        
        return LSTM_layer, dense, scaler, inputs
    
    def predict_LSTM(self,
                     inputs,
                     scaler,
                     LSTM_layer,
                     dense,
                     current_date,
                     interval,
                     periods):
        '''    
        Generates a sequence of predicted stock values.

        Arguments:
        inputs: A list containing the model's inputs [x0, a0, x0]
        scaler: Scaler that has been fitted to the data
        LSTM_layer: A trained LSTM layer
        dense: A trained dense layer
        current_date: Date after which predicitions will be made
        interval: Integer, number of days between each predicition
        periods: Integer, number of predictions

        Returns:
        forecast: Pandas series with the forecast
        '''
        args = self.args
        x0, a0, c0 = inputs
        # Instantiate the predictor
        predictor = self.predictor_model(LSTM_layer,
                                         dense,
                                         interval,
                                         periods)

        # Predictor is now fitted to data prior to current_date
        # print(x0.shape, a0.shape, c0.shape)
        pred = predictor.predict_on_batch([x0, a0, c0])
        # print(f'{x0 = }')
        # print(f'{pred = }')
        
        if args['include_dates']:
            dates = [current_date + pd.Timedelta(int(pred[p][0]), 'D') for p in range(len(pred))]
            y = [value[1] for value in pred]
        else:
            dates = [current_date + (p+1)*pd.Timedelta(interval, 'D') for p in range(len(pred))]
            y = [value[0] for value in pred]
        
        if bool(args['difference']):
            y = np.cumsum(np.concatenate(([last_price], y)))[1:] # Integrate
        
        # print(f'{y = }')
        y = scaler.inverse_transform(np.reshape(y, (-1, 1)))[:, 0]
        # print(f'{y = }')

        forecast = pd.DataFrame(data = {'y': y, 'ds': dates}).set_index('ds').squeeze()
        # print(forecast)
        
        return forecast
    
    def predictor_model(self, 
                        LSTM_layer, 
                        dense,
                        interval,
                        periods):
        '''    
        Returns a model to generate a sequence of stock price predictions.

        Arguments:
        LSTM_layer: A trained LSTM layer
        dense: A trained dense layer
        interval: Integer, number of days between each prediction
        periods: Integer, number of successive predictions

        Returns:
        predictor: The predictor model instance
        '''
        n_a = LSTM_layer.units
        
        args = self.args
        seq_len = args['seq_len']; n_features = args['n_features'] 
        
        x0 = Input(batch_shape = (1, seq_len, n_features))
        a0 = Input(batch_shape = (1, n_a), name = 'a0')
        c0 = Input(batch_shape = (1, n_a), name = 'c0')
        
        x, a, c = x0, a0, c0
        
        outputs = []
        
        day_count = 0
        
        while day_count < interval*periods:
            _, a, c = LSTM_layer(inputs = x, initial_state = [a, c])
            # print(f'{a = }')
            new_day_count = day_count + 1
            if (new_day_count+2)//7 == 0:
                new_day_count += 2 # Weekend stock prices not included in the time series training data
            output = Reshape([1, n_features])(dense(a)) # shape = (batch size, 1, n_features)
            # print(f'{output.shape = }')
            if new_day_count//interval > day_count//interval:
                outputs.append(output[0, 0]) # output is remembered every interval days
            day_count = new_day_count
            x = Concatenate(axis=1)([x[:, 1:, :], output]) 
            # print(f'{x.shape = }')
        
        predictor = Model(inputs = [x0, a0, c0], outputs = outputs)
        
        return predictor
    

class recommender():
    ''' 
    Simulates trading of stocks based on the predictions of a statistical tool.
    
    Parameters
    ----------
    initial_stock: Initial stock owned in the simulation
    max_trade: Maximum number of stock units to be traded when a recommendation is made
    intensity: The quantity of stock traded per period is proportional to this parameter
    min_delta: Minimum predicted relative variation of the stock price to 
               recommend a BUY or SELL action.
               
    '''
    def __init__(self,
                 oracle,
                 initial_stock=100, 
                 max_trade = 100, 
                 intensity = 3, 
                 min_delta = 0.05):
        self.initial_stock = initial_stock
        self.max_trade = max_trade
        self.intensity = intensity*max_trade
        self.min_delta = min_delta
        self.seasonal_order = False
        self.oracle = oracle
        
    def __call__(self, args):
        ''' 
        Performs a series of recommendations with the following procedure loop:
        
        1) Fits predictor to the data up to current_date from this iteration 
        2) Makes a prediction of the future stock price (after current_date)
            within a specified timeframe
        3) Makes a recommendation (BUY/SELL/HOLD) based on that prediction
        4) Updates current_date to the last date the prediction covered
    
        This sequence is looped up until a final date is reached.
            
        Parameters
        ----------
        history: 
            pd.DataFrame
            DataFrame with the timeseries data to learn from
        start_date:
            Minimum date up to which the predictor will be fitted
        end_date:
            Last date that the predictions extend to
        interval:
            Number of days separating each prediction point
        periods:
            Integer, predictor is re-fitted to the data up to current date 
            after every "periods" predictions
        
        Returns
        ----------
        A plot of the predictions compared to the real data, the financial
        balance resulting from applying the successive recommendations
        '''
        history, start_date, end_date, periods, interval = args
        
        ## Initiate all the relevant values
        # Basic attributes
        self.stock = self.initial_stock
        initial_price= float(history.loc[(history['ds']-start_date).abs().idxmin()]['y'])
        self.initial_value = self.initial_stock*initial_price # Value of Initial Stock
        self.gains = 0
        
        # Historical attributes
        self.value_history = [] # History of the variation of the value of stock being held
        self.gain_history = [] # History of gains resulting from each successive traded stock
        self.net_gain_history = [] # History of gains resulting from each successive traded stock 
                                    # net of that traded stock value at the next period
                                    # The net gain will be negative if the price of the stock sold increases 
                                    # or if the price of the stock bought decreases
        self.action_history = [] # History of each trading action performed
        
        # Prepare history and self.predictions
        history = history[['ds', 'y']]
        history = history.set_index('ds').squeeze()
        mask = history.index<start_date
        # forecast, prediction, next_date = self.forecast(history, start_date, periods, interval)
        # The way the predictor is fitted to the data will be displayed
        self.predictions = pd.Series(data = []) 
        # Initialization of self.predictions which will be a concatenation of forecasts
        
        # Loop predictions until end_date
        current_date = start_date
        plt.figure()
        plt.axvline(start_date, color = 'b') # Data fitted up to that point
        while current_date < end_date:
            # Fit the predictor to the history of stock prices and forecast
            print(f'{current_date = }')
            forecast, prediction, next_date = self.forecast(history, current_date, periods, interval)
            print(f'{forecast =}')
            print(f'{next_date = }')
            plt.axvline(next_date, color = 'k') # A green vertical line indicates that prediction was made at this date
            self.recommend(history, prediction, current_date, next_date)
            if self.predictions.empty:
                self.predictions = forecast
            else:
                self.predictions = pd.concat([self.predictions, forecast], ignore_index = False)
            # print(f'{self.predictions = }')
            current_date = next_date
        current_price = float(history.iloc[pd.Series(history.index-current_date).abs().idxmin()])
        
        # Next, print all the relevant historical attributes from this recommender to analyze its performance
        self.totalvalue_history = [round(a+b,1) for a,b in zip(self.gain_history, self.value_history)]
        print(f'\nThe Wallet gains per trade is (positive values = SELL)\n{list(zip(self.action_history, self.gain_history))}')
        print(f'\nThe Stock Value gains per trade is\n{list(zip(self.action_history, self.value_history))}')
        print(f'\nStock + Wallet value variation after trading: \n{list(zip(self.action_history, self.totalvalue_history))}')
        print(f'\nStock + Wallet Gains related to stock traded: \n{list(zip(self.action_history, self.net_gain_history))}')
        print(f'with sum of {round(sum(self.net_gain_history), 1)}')
        print(f'\nInitial stock (Quantity = {self.initial_stock}) value on {start_date}: {round(self.initial_value, 1)}')
        print(f'\nInitial stock (Quantity = {self.initial_stock}) value on {end_date}: {round(self.initial_stock*current_price, 1)}')
        print(f'\nFinal stock (Quantity = {round(self.stock, 3)}) value on {end_date}: {round(self.value, 1)}')
        print(f'\nGains from trading: {round(self.gains, 1)}')
        print(f'\nBalance compared to initial stock value on {end_date} (value owned if no trading) = {round(self.gains + self.value - self.initial_stock*current_price,1)}')
        print(f'\nBalance compared to initial stock value on {start_date}: {round(self.gains + self.value - self.initial_value,1)}')
        
        # Plot the predictions compared to real data
        self.MAE, self.error = self.MAE(history, self.predictions, start_date, end_date)
        print(f'\n{self.MAE = }')
        print(f'\n{self.error = }')
        rolling = history.rolling(20)
        mean, std = rolling.mean(), rolling.std()
        bolling_min = pd.Series(mean-2*std)
        bolling_max = pd.Series(mean+2*std)
        plt.title(f'Forecast and real data compared for {symbol}')
        plt.plot(self.predictions, color = 'g', label = 'Predictions')
        plt.plot(history, ':', color = 'k', label = 'History')
        plt.plot(bolling_min, color = 'b', label = 'Top of Bollinger Band')
        plt.plot(bolling_max, color = 'r', label = 'Bottom of Bollinger Band')
        plt.legend()
        plt.xlabel('Date')
        plt.xticks(fontsize = 8) 
        plt.ylabel('Open')
        plt.show()
        
    def recommend(self, history, prediction, current_date, next_date):
        '''
        Performs a trading recommendation based on the predictor's forecast.
        Updates the recommender attributes related to stock held, 
        profit made, and history of recommendations performed.
        
        Arguments
        ----------
        history: Pandas Series, history of stock prices
        prediction: Float, predicted stock price at next date
        current_date: Date up to which stock price data is learned
        next_date: Future date, at which the forecasted stock price 
                    is the basis for the trading recommendation
        
        Updates
        ----------
        self.action_history: List, list of trading actions simulated
        self.stock: Float, simulated stock quantity
        self.value: Float, simulated stock value
        self.gains: Float, simulated gains from stock bought or sold
        self.net: Float, simulated net gains from stock bought or sold
                    (compared to no trading)
        self.value_history: List, subsequent self.value values
        self.gain_history: List, subsequent self.gain values
        self.net_gain_history: List, subsequent self.net values
        '''
        current_price = float(history.iloc[pd.Series(history.index-current_date).abs().idxmin()])
        self.value = self.stock*current_price # Current stock value
        actual_next_price = float(history.iloc[pd.Series(history.index-next_date).abs().idxmin()])
        
        relative = (prediction-current_price)/current_price # Relative price change
        action = ["SELL", "BUY"]
        
        # Determine action to take based on predicted relative price change
        if abs(relative) >= self.min_delta:
            t = prediction/current_price 
            sign = int(t>1/t) - int(t<1/t) # equals 1 if prediction>current_price, else -1
            trade = round(max(-self.stock, min(self.intensity*(max(t, 1/t)-1), self.max_trade)*sign), 6)
                # The amount of stock traded (self.intensity*(max(t, 1/t)-1)) 
                # is higher when a larger price change is predicted
            self.stock += trade
            self.gains -= trade*current_price
            self.net = trade*(actual_next_price-current_price)
            self.action_history.append(action[int(relative>0)]) # BUY or SELL
        else:
            self.net = 0
            self.action_history.append("HOLD")
            trade = 0
    
        # Update value of stock being held
        self.value_history.append(round(self.stock*actual_next_price - self.value,1))
        self.value = self.stock*actual_next_price # Updated stock value
        self.gain_history.append(round(-trade*current_price,1))
        self.net_gain_history.append(round(self.net, 1))
    
    def forecast(self, 
                 history, current_date, 
                 periods, interval):
        '''
        Performs a trading recommendation based on the predictor's forecast.
        Updates the recommender attributes related to stock held, 
        profit made, and history of recommendations performed.
        
        Arguments
        ----------
        history: Pandas Series, history of stock prices
        current_date: Date up to which stock price data is learned
        periods: Integer, number of predictions
        interval: Integer, number of days between each prediction
        
        Returns
        ----------
        forecast: Pandas series, with stock price predictions
        prediction: Float, last stock price prediction
        next_date: Date at which the last stock price is predicted
        '''
        oracle = self.oracle
        forecast, prediction, next_date = oracle(history,
                                                 current_date,
                                                 periods,
                                                 interval)
        return forecast, prediction, next_date

    def MAE(self, history, forecast, start_date, end_date):
        '''Returns the Mean Absolute Error of the forecast'''
        hmask = (history.index >= start_date) & (history.index <= end_date)
        fmask = (forecast.index >= start_date) & (forecast.index <= end_date)
        error = history[hmask].subtract(forecast[fmask]).abs().dropna()
        return round(error.mean(), 1), error

print('models was run')