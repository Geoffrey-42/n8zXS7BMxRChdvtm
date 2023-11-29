# Stock Price Prediction and Recommendation System

This project aims to develop a stock price prediction and recommendation system for a portfolio investment company. The company's primary objective is to achieve sustainable returns through value investing in emerging markets worldwide. Their investment strategy focuses on long-term growth and carefully managing the investment portfolio to achieve desired results for their clients.

## Objective

The primary objective of this project is to predict stock prices on a weekly to monthly basis and provide recommendations for BUY, HOLD, or SELL decisions, with the aim of maximizing capital returns.

## Dataset

To accomplish this objective, a comprehensive dataset comprising the stock prices of multiple companies from the period of 2020 to 2023 is acquired from financial libraries using Python.

## Methodology

# Forecasting Models

The project includes a forecaster class that is capable of performing stock price forecasting based on a user-selected statistical model with customizable parameters. The covered models include AutoReg, ARIMA, SARIMA, fbprophet, and an LSTM model.

# Recommendation System

In addition to stock price forecasting, a recommender class is implemented to generate a series of trading recommendations based on the outputs of the forecaster. These recommendations provide guidance for making informed trading decisions.

# LSTM Model

After extensive hyperparameter tuning, the LSTM model emerged as the most effective in predicting stock prices. The implemented forecaster class incorporates several notable features, such as differencing the time series and activating the stateful mode in the LSTM. This allows the model to learn and utilize underlying patterns in the data, providing valuable context when analyzing temporal sequences of stock prices. The superior capabilities of the LSTM model position it as a valuable tool for making informed investment decisions.

## Usage

To use this stock price prediction and recommendation system, follow these steps:

1. Acquire the comprehensive dataset containing stock prices from financial libraries.
2. Select the desired statistical model with customizable parameters from the forecaster class.
3. Run the forecaster class to perform stock price forecasting.
4. Generate trading recommendations using the recommender class based on the forecasted prices.
5. Make informed trading decisions based on the recommendations.

## Conclusion

This stock price prediction and recommendation system provides a valuable tool for the portfolio investment company to achieve sustainable returns through value investing. By accurately predicting stock prices and providing informed trading recommendations, the system aims to maximize capital returns for the company and its clients.
