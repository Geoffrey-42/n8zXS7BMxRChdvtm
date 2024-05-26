# Stock Price Prediction and Recommendation System

This project implements multivariate financial time series forecasting and simulates the application of an informed trading strategy for a portfolio investment company.

A holistic approach is taken regarding the data collected to infer the forecasts. Specifically, the data collected is not restricted to only historical stock prices but also includes macro-economic indicators and internet trends. Significant attention is given to the feature engineering stage before performing automatic model hyperparameter tuning.

Thanks to this data-centric approach, a model featuring only an LSTM layer followed by a dense layer is sufficient to achieve profitable and accurate stock price forecasting.

## Objective

The primary objective of this project is to predict stock prices on a weekly to monthly basis and provide recommendations for BUY, HOLD, or SELL decisions, with the aim of maximizing capital returns.

## Dataset

The raw dataset used in this project can be split into three categories:
  - Historical data about the user-selected stock (Open, Close, High, Low, Volume)
  - SMIS macroeconomic indicators (^IXIC, ^GSPC, DJI), with each SMIS stock having regular (Open, Close, High, Low, Volume) data
  - Internet trends (googletrends & wikipediatrends) resulting from keywords related to the stock. For example, for Apple stock (AAPL), possible keywords could be "iPhone" or "MacBook".

This work also serves as a Proof-of-Concept (PoC) demonstrating that an approach centered around feature engineering and the construction of a more holistic dataset can be more efficient than a model-centric approach for financial time series forecasting.

## Methodology

### Feature Engineering
First, the original raw dataset is extended with a large number of technical indicators using pandas-ta, a Python library specialized in Technical Analysis.
  - This results in a total of about 260 technical features.

Second, a feature selection algorithm is run based on each feature's importance for the binary classification task of predicting whether the stock price will go up or down in 14 days.
  - A Gradient Boosting Regressor from sktime with a max_depth of 1 is utilized as a regressor. This results in about 20-30 features being selected out of the original 260.
  - The overall predictive power of the scaled selected features is then tested with the HIVECOTEV2 and Rocket state-of-the-art classifiers implemented in sktime on the same binary classification task, generally resulting in about 60-70% accuracy.

Third, the selected features are transformed using the Principal Component Analysis (PCA) algorithm implemented in sklearn, and then scaled.
  - Only the Principal Components retaining about 90% of the total variance are selected, which generally results in about 10 final engineered features.
  - When tested on the binary classification task mentioned above using HIVECOTEV2 and Rocket, the accuracy is superior to the previous test results, usually in the range of 70-80%.

The target used for the regression task that follows is the scaled Return.  
The unscaled Return is defined by:

<div align="center">Return[i] = <sup>Close[i + horizon]</sup>&frasl;<sub>Close[i]</sub> - 1</div>

where horizon is the forecast window in days, here selected to be 14 days.

### Forecasting Models

A forecaster class is defined which performs stock price forecasting based on a user-selected machine learning model with customizable parameters.
This allows flexibility in defining what model to use while using the same framework to generate the forecasts.
By default, an instance of the forecaster class features a simple LSTM model comprising an LSTM layer followed by a dense layer.

### Recommendation System

A recommender class is implemented to simulate the application of a trading strategy over a certain period based on the outputs of the forecaster instance.
The goal is to verify if the trading policy combined with the accuracy of the forecasting model results in a long-term profit for the investor.

### Automatic Hyperparameter Tuning with Optuna

Optuna is a Python library that implements state-of-the-art Bayesian optimization for hyperparameter tuning of any model.
Moreover, the results can be saved in a log file or database to track the outcomes. The library also provides advanced visualization capabilities to analyze the hyperparameter search with optuna-dashboard.

Thanks to this approach, the regressor model's hyperparameters are tuned automatically, their relative importances computed, and their relationships analyzed.
Eventually, optimal hyperparameters can be derived.

## Results

Using the engineered features and optimal model hyperparameters, stocks are forecasted and trades are simulated over a period of time following those of the training and tuning stages.
The forecasted prices are generally accurate, and the trading simulations result in a profitable outcome. This proves that the engineered features carried sufficient predictive power and were formatted correctly to be utilized by an LSTM model.

More details on the results can be found in the notebooks, with the forecasting of Apple, Amazon, Microsoft, and Meta stock prices, as well as Bitcoin.

## Usage

To use this stock price prediction and recommendation system, clone this repo and follow these steps:

  - Choose a stock to forecast (e.g., stock_name = 'AAPL')
  - If the stock chosen is not already covered by this repo (if the notebook does not exist):
      - Download its historical data (Open, Close, High, Low, Volume) and save it as a pickle file in the data folder. This can be done using the extract_financial_data function with online=True and save=True arguments (refer to a notebook for usage example).
      - Download googletrends and wikipediatrends CSV files related to keywords relevant to the stock and place them in the data folder. Ensure the names follow the formats "googletrend-chart_{keyword}_{searchtype}.csv" and "wikishark-chart_{keyword}[en].csv". The CSV files data should be in the as downloaded format.
  - Choose a model supported by the forecaster (e.g., predictor_name = 'LSTM')
  - Set up all the arguments (model, training, temporal, data, and trading arguments) or leave them as default. Refer to a notebook for guidance.
  - Choose whether to perform a hyperparameter search (hyperparameter_search = False/True)
  - Create a forecaster instance and a recommender instance (refer to the notebook)
  - Call the recommender instance to simulate a series of forecasts and trading actions every 14 days between start_date and end_date.
  - Visualize the results on the generated graph and check the printed logs for more details on performance.

## Conclusion

One of the main findings from this work is that multivariate time series forecasting, which leverages not only stock market data but also external features such as googletrends, is more efficient than an analysis based solely on stock market data. This shows that a data-centric approach with appropriate feature selection and engineering can yield better results than a model-centric approach with complex hybrid models.
