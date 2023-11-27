The portfolio investment company is an external client with a primary objective to achieve sustainable returns through value investing in emerging markets worldwide. The company aims to identify profitable companies based on their intrinsic value and make strategic decisions accordingly. The investment strategy focuses on long-term growth, avoiding frequent trading based on short-term market fluctuations. Their goal is to carefully select and manage the investment portfolio to achieve the desired results for the client. 

The primary objective of this project is to predict stock prices on a weekly to monthly basis and provide recommendations for BUY, HOLD, or SELL decisions, with the aim of maximizing capital returns.

In order to accomplish this, a comprehensive dataset encompassing the stock prices of multiple companies from the period of 2020 to 2023 is acquired from financial libraries with Python.

The forecaster class developed as part of the project has the capability to perform stock price forecasting based on a user-selected statistical model with customizable parameters. The covered models included AutoReg, ARIMA, SARIMA, fbprophet, and an LSTM model.

Additionally, a recommender class is implemented to generate a series of trading recommendations based on the outputs of the forecaster. These recommendations provide guidance for making informed trading decisions.

After extensive hyperparameter tuning, the LSTM model emerged as the most effective in predicting stock prices. The implemented forecaster class incorporated several notable features, including the ability to difference the time series and activate the stateful mode in the LSTM. This allowed the model to learn and utilize the underlying patterns in the data, providing valuable context when analyzing temporal sequences of stock prices. The superior capabilities of the LSTM model position it as a valuable tool for making informed investment decisions.
