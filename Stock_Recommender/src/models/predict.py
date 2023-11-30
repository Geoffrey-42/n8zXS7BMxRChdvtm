runfile('src/models/models.py')

# Choose what to forecast
symbols = ["BTC-USD", "GOOG", "MSFT", "KCHOL.IS", "BEEF3.SA", "PAM", "CMTOY", "IMP.JO"]
symbol = symbols[2]
history = hist[symbol]

# Choose the model
predictors = ['LSTM', 'Prophet', 'AutoReg', 'ARIMA', 'SARIMAX', 'SimpleExpSmoothing']
predictor = predictors[0]

# Set the model parameters
args = defaultdict(int)

arguments = {'seq_len': 120,
             'n_features': 1,
             'learning_rate': 0.008,
             'loss': 'huber',
             'decay': 0.99,
             'epochs': 20,
             'n_a': 64,
             'include_dates': False,
             'stateful': True,
             'difference': False}

for key, value in arguments.items():
    args[key] = value

# Set the time parameters
start_date = pd.to_datetime('2022-09-01')
end_date = pd.to_datetime('2023-12-01')
periods = 2 # prediction range
interval = 7 # in days

# Set the trading parameters
initial_stock = 10
max_trade = 10
intensity = 3 # Price variation by 1/intensity results in trading max_trade
min_delta = 0.05

# Create a forecaster object
clairvoyant = forecaster('LSTM', args)

# Create a recommender object
recommend = recommender(oracle = clairvoyant,
                        initial_stock = initial_stock, 
                        max_trade = max_trade,
                        intensity = intensity,
                        min_delta = min_delta)

# Perform forecasting and recommendations
args = (history, start_date, end_date, periods, interval)
recommend(args) # performs the recommendation