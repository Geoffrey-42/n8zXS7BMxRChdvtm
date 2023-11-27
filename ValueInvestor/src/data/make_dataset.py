runfile('setup.py')
dir_path = os.getcwd()
dir_path = dir_path.replace('\\', '/')

excel_data = pd.read_excel(dir_path + '/data/2020Q1Q2Q3Q4-2021Q1.xlsx',
                           sheet_name = None, 
                           parse_dates = True)
# The data size is small, a larger time frame will be downloaded
names = list(excel_data.keys()) 
symbols = {"MSFT", "SBER.ME", "KCHOL.IS", "BEEF3.SA", "PAM", "CMTOY", "IMP.JO", "460860.KS"}
symbols = {"BTC-USD", "GOOG", "MSFT", "SBER.ME", "KCHOL.IS", "BEEF3.SA", "PAM", "CMTOY", "IMP.JO"}

# Pull the data from January 2020 to current date
tickers = yf.Tickers(" ".join(symbols))

# Structure 1: Create a dictionary of Dataframes
hist = {}
for symbol in symbols:
    history = tickers.tickers[symbol].history(start="2020-01-01")
    history.rename(columns={'Open': 'y'}, inplace=True)
    history.index.rename('ds', inplace=True)
    history.reset_index(inplace=True)
    history['ds'] = history['ds'].dt.tz_localize(None)
    hist[symbol] = history
    # hist[symbol] = tickers.tickers[symbol].history(period="5y")
    # hist[symbol] = hist[symbol].loc[hist[symbol].index >=pd.to_datetime('2020-01-01T01:00:00.000000', utc=True)]

# Structure 2: One single Dataframe grouped by tickers
hist2 = yf.download(" ".join(symbols), start="2020-01-01", group_by='tickers')

# sns.distplot(hist['CMTOY']['Open'], color = 'b')
# sns.violinplot(y = 'Open', x = 'Dividends', data = hist['CMTOY'])
# sns.countplot(x = 'Dividends', data = hist['CMTOY'])
# sns.catplot('Open', 'Volume', data = hist['CMTOY'], hue = 'Dividends', height = 20)

print('make_dataset was run')