# Stock Price Prediction and Recommendation System

This work implements multivariate financial time series forecasting and simulates the application of an informed trading strategy for a portfolio investment company.  

A holistic approach is taken with regard to the data collected to infer the forecasts. 
More particularly, the data collected is not restricted to only the historical stock prices, but also macro-economic indicators and internet trends. 
A strong attention is given to the feature engineering stage, before performing an automatic model hyperparameter tuning.  

Thanks to this data-centric approach, a model only featuring a LSTM layer followed by a dense layer is sufficient to perform profitable and accurate stock price forecasting.

## Objective

The primary objective of this project is to predict stock prices on a weekly to monthly basis and provide recommendations for BUY, HOLD, or SELL decisions, with the aim of maximizing capital returns.

## Dataset

The raw dataset used in this project can split into three categories:
  - Historical data about the user-selected stock (Open, Close, High, Low, Volume)
  - SMIS macroeconomic indicators (^IXIC, ^GSPC, DJI), with for each SMIS stock the regular (Open, Close, High, Low, Volume) data
  - Internet trends (googletrends & wikipediatrends) resulting from keywords related to the stock. For example, for the Apple stock (AAPL), a possible keyword could be "Iphone" or "Macbook".

This work can also be seen as a Proof-of-Concept (PoC) that an approach centered around feature engineering and the construction of a more holistic dataset can be more efficient than a model-centric approach for financial time series forecasting.

## Methodology

### Feature Engineering
First, the original raw dataset is extended with a large number of technical indicators using pandas-ta, a python library specialized in Technical Analysis.
  - This results in a total of about 260 technical features.  

Second, a feature selection algorithm is run on the basis of each feature importance for the binary classification task of predicting whether the stock price will go up or down in 14 days.  
  - A Gradient Boosting Regressor from sktime with a max_depth of 1 is utilized as a regressor. This results in about 20-30 features being selected out of the original 260.  
  - The overall predictive power of the scaled selected features is then tested with the HIVECOTEV2 and Rocket state-of-the-art classifiers implemented on sktime on the same binary classification task, which generally results in about 60-70% accuracy.  

Third, the selected features are transformed using the Principal Component Algorithm (PCA) implemented on sklearn, and then scaled. 
  - Only the Principal Components retaining about 90% of the total variance are selected, which generally results in a total of about 10 final engineered features.  
  - When tested of the binary classification task mentionned above using HIVECOTEV2 and Rocket, the accuracy is superior to the previous test results, and are usually in the range 70-80%.

The target used for the regression task that will follows is the scaled Return.  
The unscaled Return is defined by: Return[i] = Close[i+horizon]/Close[i] - 1 where horizon is the forecast window in days, here selected to be 14 days.

### Forecasting Models

A forecaster class is defined which performs stock price forecasting based on a user-selected machine learning model with customizable parameters.
This allows for some flexibility in defining what model to use, and yet using the same framework to generate the forecasts. 
By default, an instance of the forecaster class features a simple LSTM model comprising a LSTM layer followed by a dense layer.

### Recommendation System

A recommender class is implemented to simulate the application of a trading strategy over a certain time period on the basis of the outputs of the forecaster instance.
The goal is to verify if the trading policy combined with the accuracy of the forecasting model does result in a long-term profit for the investor.

### Automatic Hyperparameter Tuning with Optuna

Optuna is a python library that implements state-of-the-art bayesian optimization for the hyperparameter tuning of any model.
Moreover the results can be saved in a log file or database to track the results. The library also provides advanced visualization capabilities to analyze the hyperparameter search with optuna-dashboard.  

Thanks to this approach, the regressor model's hyperparameters are tuned automatically, their relative importances computed and their relationship can be analyzed.
Eventually, optimal hyperparameter can be derived.

## Results

Using the engineered features and optimal model hyperparameters, stocks are forecasted and trades are simulated over a period of time posterior to the training and tuning stages.
The forecasted prices are generally accurate and the trading simulations result in a profitable outcome. This proves that the engineered features carried sufficient predictive power and were formatted in the right way to be utilized by a LSTM model.  

More details on the results can be found in the notebooks, with the forecasting of Apple, Amazon, Microsoft and Meta stock prices, as well as the Bitcoin.

## Usage

To use this stock price prediction and recommendation system, clone this repo and follow these steps:

  - Choose a stock to forecast (i.e. stock_name = 'AAPL')
  - If the stock chosen is not already covered by this repo (if notebook does not exists):
      - Download its historical data (Open, Close, High, Low, Volume) and save it as a pickle file in the data folder. This can be done using the extract_financial_data function with online=True and save=True arguments passed (refer to a notebook for usage example).
      - Download googletrends and wikipediatrends csv files relative to keywords relevant to the stock and place it in the data folder. Make sure the names follow the formats "googletrend-chart_{keyword}_{searchtype}.csv" and "wikishark-chart_{keyword}[en].csv". The csv files data should be in the as downloaded format.
  - Choose a model supported by the forecaster (i.e. predictor_name = 'LSTM')
  - Set up all the arguments (model, training, temporal, data and trading arguments) or let it as is. Refer to a notebook for guidance.
  - Choose whether to perform a hyperparameter search (hyperparameter_search = False/True)
  - Create a forecaster instance and a recommender instance (Cf. Notebook)
  - Call the recommender instance to simulate a series of forecasts and trading action every 14 days between start_date and end_date.
  - Visualize the results on the generated graph and look at the printed logs for more details on the performance.

## Conclusion

One of the main findings derived from this work is that multivariate time series forecasting that leverages not only stock market data but also external features such as googletrends is a more efficient approach than an analysis based only on stock market data. This shows that a data centric approach with approriate feature selection and engineering can bring more results than a model centric approach with complex hybrid models.
