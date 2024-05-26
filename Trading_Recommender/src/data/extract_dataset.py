import pandas as pd
import yfinance as yf
import os


def extract_financial_data(data_dir = 'data', save=False, online=False):
    '''
    For a set of stocks, extract three categories of data for financial forecasting:
        1) Stock market data (e.g. AAPL) with (Open, Close, High, Low, Volume)
        2) The SMIS macroeconomic indicators ('DJI', '^IXIC', '^GSPC')
        3) Internet trends about keywords associated with each stock
    
    The stocks involved are defined by the symbols set.
    
    Input
    ----------
    data_dir: String
        Folder where the data is located (for offline extraction)
        Folder where to save the downloaded data (for online extraction)
    
    save: Boolean
        Whether to save the downloaded data in data_dir (for online extraction)
    
    online: Boolean
        Whether to get the data online with yfinance or to get it from data_dir
    
    Output
    ----------
    stock_dict: Dictionary of DataFrame
        Key is the stock name, value is the stock market data
        
    smis_dict: Dictionary of DataFrames
        Key is the SMIS stock name, value is the associated data
        
    trend_dict: Dictionary of dictionaries of DataFrames
        Key is the stock name, value is a subdictionary with:
            SubKey as the type of internet trend (googletrend or wikipediatrend)
            Subvalue is a DataFrame in which each column is a specific keyword
    '''
    
    ## 1) Stock Market Data
    symbols = {"BTC-USD", "AAPL", "MSFT", "AMZN", "META"}
    
    if online:
        # Pull the data from yahoo finance
        stock_tickers = yf.Tickers(" ".join(symbols)).tickers
        # Structure 1: Create a dictionary of Dataframes
        stock_dict = {}
        for symbol in symbols:
            Ticker = stock_tickers[symbol]
            history = Ticker.history(start="2020-01-01",
                                     end="2024-04-30")
            history.index.rename('DateTime', inplace=True)
            history.reset_index(inplace=True)
            history['DateTime'] = history['DateTime'].dt.tz_localize(None)
            history.set_index('DateTime', inplace = True)
            stock_dict[symbol] = history
        
        # Save the downloaded files (if save=True)
        if save:
            for key, df in stock_dict.items():
                df.to_pickle(os.path.join(data_dir, f'{key}.pkl'))
    else:
        stock_dict = {}
        for symbol in symbols:
            df = pd.read_pickle(os.path.join(data_dir, f'{symbol}.pkl'))
            stock_dict[symbol] = df
    
    ## 2) Security Market Indicator Series (SMIS)
    SMIS = ['DJI', '^IXIC', '^GSPC']
    # = [Dow Jones Industrial Average, Nasdaq Composite, S&P 500]
    if online:
        # Pull the data from yahoo finance
        SMIS_tickers = yf.Tickers(" ".join(SMIS)).tickers
        # Structure 1: Create a dictionary of Dataframes
        smis_dict = {}
        for symbol in SMIS:
            Ticker = SMIS_tickers[symbol]
            history = Ticker.history(start="2020-01-01",
                                     end="2024-04-30")
            history.index.rename('DateTime', inplace=True)
            history.reset_index(inplace=True)
            history['DateTime'] = history['DateTime'].dt.tz_localize(None)
            history.set_index('DateTime', inplace = True)
            smis_dict[symbol] = history
        
        # Save the downloaded files (if save=True)
        if save:
            for key, df in smis_dict.items():
                df.to_pickle(os.path.join(data_dir, f'{key}.pkl'))
    else:
        smis_dict = {}
        for symbol in SMIS:
            df = pd.read_pickle(os.path.join(data_dir, f'{symbol}.pkl'))
            smis_dict[symbol] = df
    
    ## 3) Wikipedia and Google Trends
    trends = {
        'MSFT': {'wiki': ['Microsoft'],
                 'google': ['Microsoft_news']},
        'BTC-USD': {'wiki': ['Bitcoin'],
                    'google': ['Bitcoin_news']},
        'AAPL': {'wiki': ['Apple_Inc.', 'MacBook', 'Iphone'],
                 'google': ['Apple_news', 'MacBook_web', 'iphone_shopping']},
        'AMZN': {'wiki': ['Amazon.com', 'Amazon_Prime'],
                 'google': ['AmazonPrime_news']},
        'META': {'wiki': ['Instagram'],
                 'google': ['Instagram_news', 'Meta_news']}
        }
    trend_dict = {}
    for symbol, trend in trends.items():
        # Form the google trends dataframe
        column_names = trend['google']
        gtrend_list = []
        for column_name in column_names:
            df = pd.read_csv(f'{data_dir}\googletrend-chart_{column_name}.csv',
                             skiprows=3, header=None)
            df.columns = ['DateTime'] + [f'googletrends_{column_name}']
            df.set_index('DateTime', inplace=True)
            df.index = pd.to_datetime(df.index)
            gtrend_list.append(df)
        gtrends = pd.concat(gtrend_list, axis=1)
        trend_dict[symbol] = {'googletrends': gtrends}
        # Form the wikipedia trends dataframe
        column_names = trend['wiki']
        wtrend_list = []
        start_date = pd.to_datetime("2020-01-01")
        end_date = pd.to_datetime("2024-04-30")
        for column_name in column_names:
            df = pd.read_csv(f'{data_dir}\wikishark-chart_{column_name}[en].csv',
                             sep=';', skiprows=1, index_col=0, header=None)
            df.columns = [f'wikitrends_{column_name}']
            df.index = pd.to_datetime(df.index)
            df.index.name = 'DateTime'
            mask = (df.index >= start_date) & (df.index <= end_date)
            wtrend_list.append(df[mask])
        wtrends = pd.concat(wtrend_list, axis = 1)
        trend_dict[symbol]['wikitrends'] = wtrends
    
    ## 5) Return the stock market data, the SMIS data and the web trends
    return stock_dict, smis_dict, trend_dict

if (__name__  == '__main__'):
    stock_dict, smis_dict, trend_dict = extract_financial_data(data_dir = '../../data', 
                                                               save=False, online=False)
    # import seaborn as sns
    # sns.displot(history['BTC-USD']['Open'], color = 'b')
    # sns.histplot(history['BTC-USD']['Open'], color = 'b')
    # sns.violinplot(y = 'Volume', x = 'gtrend', data = history['BTC-USD'])
    # sns.violinplot(y = 'Close', x = 'gtrend', data = history['BTC-USD'])
    # sns.countplot(x = 'gtrend', data = history['BTC-USD'])
    # sns.catplot(x='gtrend', y='Close', data = history['BTC-USD'], hue = 'Volume', height = 20)
