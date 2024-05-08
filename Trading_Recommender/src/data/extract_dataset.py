import pandas as pd
import pandas as pd
import yfinance as yf
import seaborn as sns
import os


def extract_financial_data(data_dir = '', save=False, online=False):
    if not(online):
        save = False
    symbols = {"MSFT", "SBER.ME", "KCHOL.IS", "PAM", "CMTOY", "IMP.JO", "460860.KS"}
    symbols = {"BTC-USD", "GOOG", "MSFT", "SBER.ME","BEEF3.SA", "KCHOL.IS", "PAM", "CMTOY", "IMP.JO"}
    
    # Pull the data from January 2020 to current date
    tickers = yf.Tickers(" ".join(symbols))
    
    if online:
        # Structure 1: Create a dictionary of Dataframes
        hist = {}
        for symbol in symbols:
            history = tickers.tickers[symbol].history(start="2020-01-01")
            history.rename(columns={'Open': 'y'}, inplace=True)
            history.index.rename('ds', inplace=True)
            history.reset_index(inplace=True)
            history['ds'] = history['ds'].dt.tz_localize(None)
            history.set_index('ds', inplace = True)
            hist[symbol] = history
            # hist[symbol] = tickers.tickers[symbol].history(period="5y")
            # hist[symbol] = hist[symbol].loc[hist[symbol].index >=pd.to_datetime('2020-01-01T01:00:00.000000', utc=True)]
        
        # Structure 2: One single Dataframe grouped by tickers
        hist2 = yf.download(" ".join(symbols), start="2020-01-01", group_by='tickers')
    else:
        hist = {}
        for symbol in symbols:
            df = pd.read_pickle(os.path.join(data_dir, f'{symbol}.pkl'))
            hist[symbol] = df
    
    if save:
        for key, df in hist.items():
            df.to_pickle(os.path.join(data_dir, f'{key}.pkl'))
    
    return hist

if (__name__  == '__main__'):
    hist = extract_financial_data(save=False, online=False)
    sns.displot(hist['CMTOY']['y'], color = 'b')
    sns.histplot(hist['CMTOY']['y'], color = 'b')
    sns.violinplot(y = 'y', x = 'Dividends', data = hist['CMTOY'])
    sns.violinplot(y = 'y', x = 'Close', data = hist['CMTOY'])
    sns.countplot(x = 'Dividends', data = hist['CMTOY'])
    sns.catplot(x='y', y='Close', data = hist['CMTOY'], hue = 'Dividends', height = 20)
