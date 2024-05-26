from src.data.bars import Bars # from https://github.com/JonasHane2/Bars
from itertools import product
import matplotlib.pyplot as plt

# Taken from https://ved933409.medium.com/major-reasons-why-machine-learning-fails-in-stock-prediction-part-02-7cc201305ce2
'''
Continuing from the previous article where I was discussing the mistake made 
by many practitioners and academic papers of inefficient sampling of data.
We will discuss some more advanced ways of sampling the stock price data.

In a market equilibrium condition, buyers and sellers are equally active 
in the market. The primary objective of an information-driven bar is to gather 
information when either buyers or sellers become more active than the other. 
This indicates that more informed traders are entering the market, and we may 
be able to make decisions before the market reaches its equilibrium level.

In this section, we will look into different indices of information arrival.
'''

'''
time-sampled [financial] series often exhibit poor
statistical properties, like serial correlation, heteroscedasticity, and non-normality of
returns (Easley, L´opez de Prado, and O’Hara [2012]).
'''

def sample_features(features,
                    target_samples_per_day = 1/14,
                    target = 'Close',
                    bar_type = 'dollar',
                    imbalance_sign = False,
                    beta = 10000):
    selected_columns = features[['Volume', target]].copy()
    selected_columns.rename(columns={target: 'Price'}, inplace=True)
    bars = Bars(bar_type, imbalance_sign, target_samples_per_day, beta)
    dates = bars.get_all_bar_ids(selected_columns)
    return features.loc[dates]

def plot_bars(features):
    # Taken from the notebook in https://github.com/JonasHane2/Bars
    selected_columns = features[['Volume', 'Close']].copy()
    selected_columns.rename(columns={'Close': 'Price'}, inplace=True)
    target_samples_per_day = 1/14
    beta = 10000
    bar_types = ['tick', 'volume', 'dollar']
    imbalance_signs = [False, True]
    id_bars = []
    for imbalance_sign, bar_type in product(imbalance_signs, bar_types):
        bars = Bars(bar_type, imbalance_sign, target_samples_per_day, beta)
        temp = bars.get_all_bar_ids(selected_columns)
        id_bars.append(temp)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig.suptitle('Price Line with Dots',
                 fontsize = 21)
    
    for i, (ax, (imbalance_sign, bar_type)) in enumerate(zip(axes.flatten(), product(imbalance_signs, bar_types))):    
        ax.plot(selected_columns['Price'], label='Price Line')
        ax.scatter(id_bars[i], selected_columns['Price'].loc[id_bars[i]], label=f'{bar_type}{" Imbalance" if imbalance_sign else ""} Bars', color='red')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{bar_type}{" Imbalance" if imbalance_sign else ""} Bars')
        ax.legend()

if (__name__ == "__main__"):
    from bars import Bars
    ## Assuming 9 pkl files are present in the {project_path}/data folder
    ## Otherwise change the plotting parameters
    from extract_dataset import extract_financial_data
    hist = extract_financial_data(data_dir = '../../data', save=False, online=False)
    # One example
    features = hist['BTC-USD']
    # Sampling parameters
    target_samples_per_day = 1/14
    imbalance_sign = False
    bar_type = 'dollar'
    beta = 10000
    # Plotting all possibles bars
    plot_bars(features)
    # Now selecting a bar, sampling all stock data
    sampled_histories = []
    for symbol in hist.keys():
        features = hist[symbol]
        sampled_histories.append(sample_features(features,
                                                target_samples_per_day = 1/14,
                                                imbalance_sign = False,
                                                bar_type = 'dollar',
                                                beta = 10000))
    # Plotting the result
    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize=(15, 10))
    fig.subplots_adjust(hspace = 0.4, wspace = 0.2)
    fig.suptitle(f'Stock Prices: Time Bars VS {bar_type}{" Imbalance" if imbalance_sign else ""} Bars',
                 fontsize = 21)
    for i, (symbol, ax) in enumerate(zip(hist.keys(), axes.flatten())):
        features = hist[symbol]
        sampled_features = sampled_histories[i]
        ax.scatter(sampled_features['Close'].index, sampled_features['Close'], color = 'r', label = 'sampled features', marker = 'x')
        ax.plot(features['Close'], color = 'navy', label = 'features', linewidth = 0.7)
        ax.legend(fontsize = 'medium')
        ax.tick_params(axis = 'x', labelsize = 8)
        x = features.index
        ax.set_xticks(x[::len(x)//4])
        #ax.set_xticklabels(x[::len(x)//4])
        ax.set_xlabel('Date')
        ax.set_ylabel('Close')
        ax.set_title(f'{symbol} price ({len(sampled_features)}/{len(features)} subsampled)', fontsize = 12)
    # plt.show()
