import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.sample_dataset import sample_features

def compute_technical_indicators(stock_extended, smis_dict, trends):
    '''
    Compute the technical indicators for the stock market data, the SMIS data
    and the internet trends.
    
    Inputs
    ----------
    stock_extended: Pandas DataFrame
        The stock market data that will be extended with technical features
    smis_dict: Dictionary
        Contains pandas dataframes with macroeconomics indicators
    trends: Dictionary
        Contains information about internet trends
    
    Outputs
    ----------
    stock_extended: Pandas DataFrame
        The time series extended with technical features
    smis_extended: Dictionary of DataFrames
        Macroeconomics indicators extended with technical features
    trend_extended: Dictionary of DataFrames
        Internet trends extended with technical features
    '''
    # Momentum (41)
    stock_extended.ta.ao(append=True) # Awesome Oscillator (AO)
    stock_extended.ta.apo(append=True) # Absolute Price Oscillator (APO)
    stock_extended.ta.bias(append=True) # Bias
    stock_extended.ta.bop(append=True) # Balance of Power (BOP)
    stock_extended.ta.brar(append=True) # BRAR
    stock_extended.ta.cci(append=True) # Commodity Channel Index (CCI)
    stock_extended.ta.cfo(append=True) # Chande Forecast Oscillator (CFO)
    stock_extended.ta.cg(append=True) # Center of Gravity (CG)
    stock_extended.ta.cmo(append=True) # Chande Momentum Oscillator (CMO)
    stock_extended.ta.coppock(append=True) # Coppock Curve
    stock_extended.ta.cti(append=True) # Correlation Trend Indicator (CTI)
    stock_extended.ta.dm(append=True) # Directional Movement (DM)
    stock_extended.ta.er(append=True) # Efficiency Ratio (ER)
    stock_extended.ta.eri(append=True) # Elder Ray Index (ERI)
    stock_extended.ta.fisher(append=True) # Fisher Transform
    stock_extended.ta.inertia(append=True) # Inertia
    stock_extended.ta.kdj(append=True) # KDJ
    stock_extended.ta.kst(append=True) # KST Oscillator
    stock_extended.ta.macd(append=True) # Moving Average Convergence Divergence (MACD)
    stock_extended.ta.mom(append=True) # Momentum
    stock_extended.ta.pgo(append=True) # Pretty Good Oscillator (PGO)
    stock_extended.ta.ppo(append=True) # Percentage Price Oscillator (PPO)
    stock_extended.ta.psl(append=True) # Psychological Line (PSL)
    stock_extended.ta.pvo(append=True) # Percentage Volume Oscillator (PVO)
    stock_extended.ta.qqe(append=True) # Quantitative Qualitative Estimation (QQE)
    stock_extended.ta.roc(append=True) # Rate of Change (ROC)
    stock_extended.ta.rsi(append=True) # Relative Strength Index (RSI)
    stock_extended.ta.rsx(append=True) # Relative Strength Xtra (RSX)
    stock_extended.ta.rvgi(append=True) # Relative Vigor Index (RVGI)
    stock_extended.ta.stc(append=True) # Schaff Trend Cycle (STC)
    stock_extended.ta.slope(append=True) # Slope
    stock_extended.ta.smi(append=True) # SMI Ergodic (SMI)
    stock_extended.ta.squeeze(append=True) # Squeeze
    stock_extended.ta.squeeze_pro(append=True) # Squeeze Pro
    stock_extended.ta.stoch(append=True) # Stochastic Oscillator (STOCH)
    stock_extended.ta.stochrsi(append=True) # Stochastic RSI (STOCHRSI)
    # stock_extended.ta.td_seq(append=True) # TD Sequential (TD SEQ)
    stock_extended.ta.trix(append=True) # Trix
    stock_extended.ta.tsi(append=True) # True strength index (TSI)
    stock_extended.ta.uo(append=True) # Ultimate Oscillator (UO)
    stock_extended.ta.willr(append=True) # Williams %R (WILLR)
    
    # Volume (15)
    stock_extended.ta.ad(append=True) # Accumulation/Distribution Index (AD)
    stock_extended.ta.adosc(append=True) # Accumulation/Distribution Oscillator (ADOSC)
    stock_extended.ta.aobv(append=True) # Archer On-Balance Volume (AOBV)
    stock_extended.ta.cmf(append=True) # Chaikin Money Flow (CMF)
    stock_extended.ta.efi(append=True) # Elder's Force Index (EFI)
    stock_extended.ta.eom(append=True) # Ease of Movement (EOM)
    stock_extended.ta.kvo(append=True) # Klinger Volume Oscillator (KVO)
    stock_extended.ta.mfi(append=True) # Money Flow Index (MFI)
    stock_extended.ta.nvi(append=True) # Negative Volume Index (NVI)
    stock_extended.ta.obv(append=True) # On-Balance Volume (OBV)
    stock_extended.ta.pvi(append=True) # Positive Volume Index (PVI)
    stock_extended.ta.pvol(append=True) # Price-Volume (PVOL)
    stock_extended.ta.pvr(append=True) # Price Volume Rank (PVR)
    stock_extended.ta.pvt(append=True) # Price Volume Trend (PVT)
    # stock_extended.ta.vp(append=True) # Volume Profile (VP)

    # Volatility (14)
    stock_extended.ta.aberration(append=True) # Aberration
    stock_extended.ta.accbands(append=True) # Acceleration Bands (ACCBANDS)
    stock_extended.ta.atr(append=True) # Average True Range (ATR)
    stock_extended.ta.bbands(append=True) # Bollinger Bands (BBANDS)
    stock_extended.ta.donchian(append=True) # Donchian Channel (DONCHIAN)
    stock_extended.ta.hwc(append=True) # Holt-Winter Channel (HWC)
    stock_extended.ta.kc(append=True) # Keltner Channel (KC)
    stock_extended.ta.massi(append=True) # Mass Index (MASSI)
    stock_extended.ta.natr(append=True) # Normalized Average True Range (NATR)
    stock_extended.ta.pdist(append=True) # Price Distance (PDIST)
    stock_extended.ta.rvi(append=True) # Relative Volatility Index (RVI)
    stock_extended.ta.thermo(append=True) # Elder's Thermometer (THERMO)
    stock_extended.ta.true_range(append=True) # True Range (TRUE_RANGE)
    stock_extended.ta.ui(append=True) # Ulcer Index (UI)
    
    # Trend (18)
    stock_extended.ta.adx(append=True) # Average Directional Movement Index (ADX)
    stock_extended.ta.amat(append=True) # Archer Moving Averages Trends (AMAT)
    stock_extended.ta.aroon(append=True) # Aroon & Aroon Oscillator (AROON)
    stock_extended.ta.chop(append=True) # Choppiness Index (CHOP)
    stock_extended.ta.cksp(append=True) # Chande Kroll Stop (CKSP)
    stock_extended.ta.decay(append=True) # Decay
    stock_extended.ta.decreasing(append=True) # Decreasing
    stock_extended.ta.dpo(append=True) # Detrended Price Oscillator (DPO)
    stock_extended.ta.increasing(append=True) # Increasing
    stock_extended.ta.long_run(append=True) # Long Run
    stock_extended.ta.psar(append=True) # Parabolic Stop and Reverse (PSAR)
    stock_extended.ta.qstick(append=True) # Q Stick (QSTICK)
    stock_extended.ta.short_run(append=True) # Short Run
    stock_extended.ta.tsignals(append=True) # Trend Signals (TSIGNALS)
    stock_extended.ta.ttm_trend(append=True) # TTM Trend (TTM_TREND)
    stock_extended.ta.vhf(append=True) # Vertical Horizontal Filter (VHF)
    stock_extended.ta.vortex(append=True) # Vortex (VORTEX)
    stock_extended.ta.xsignals(append=True) # Cross Signals (XSIGNALS)

    # Overlap (33)
    stock_extended.ta.alma(append=True) # Arnaud Legoux Moving Average (ALMA)
    stock_extended.ta.dema(append=True) # Double Exponential Moving Average (DEMA)
    stock_extended.ta.ema(append=True) # Exponential Moving Average (EMA)
    stock_extended.ta.fwma(append=True) # Fibonacci's Weighted Moving Average (FWMA)
    stock_extended.ta.hilo(append=True) # Gann High-Low Activator (HILO)
    stock_extended.ta.hl2(append=True) # High-Low Average (HL2)
    stock_extended.ta.hlc3(append=True) # High-Low-Close Average (HLC3)
    stock_extended.ta.hma(append=True) # Hull Exponential Moving Average (HMA)
    stock_extended.ta.hwma(append=True) # Holt-Winter Moving Average (HWMA)
    stock_extended.ta.ichimoku(append=True) # Ichimoku Kinkō Hyō (ICHIMOKU)
    stock_extended.ta.jma(append=True) # Jurik Moving Average (JMA)
    stock_extended.ta.kama(append=True) # Kaufman's Adaptive Moving Average (KAMA)
    stock_extended.ta.linreg(append=True) # Linear Regression (LINREG)
    # stock_extended.ta.mcgd(append=True) # McGinley Dynamic (MCGD)
    stock_extended.ta.midpoint(append=True) # Midpoint (MIDPOINT)
    stock_extended.ta.midprice(append=True) # Midprice (MIDPRICE)
    stock_extended.ta.ohlc4(append=True) # Open-High-Low-Close Average (OHLC4)
    stock_extended.ta.pwma(append=True) # Pascal's Weighted Moving Average (PWMA)
    stock_extended.ta.rma(append=True) # WildeR's Moving Average (RMA)
    stock_extended.ta.sinwma(append=True) # Sine Weighted Moving Average (SINWMA)
    stock_extended.ta.sma(append=True) # Simple Moving Average (SMA)
    stock_extended.ta.ssf(append=True) # Ehler's Super Smoother Filter (SSF)
    stock_extended.ta.supertrend(append=True) # Supertrend (SUPERTREND)
    stock_extended.ta.swma(append=True) # Symmetric Weighted Moving Average (SWMA)
    stock_extended.ta.t3(append=True) # T3 Moving Average (T3)
    stock_extended.ta.tema(append=True) # Triple Exponential Moving Average (TEMA)
    stock_extended.ta.trima(append=True) # Triangular Moving Average (TRIMA)
    # stock_extended.ta.vidya(append=True) # Variable Index Dynamic Average (VIDYA)
    stock_extended.ta.vwap(append=True) # Volume Weighted Average Price (VWAP)
    stock_extended.ta.vwma(append=True) # Volume Weighted Moving Average (VWMA)
    stock_extended.ta.wcp(append=True) # Weighted Closing Price (WCP)
    stock_extended.ta.wma(append=True) # Weighted Moving Average (WMA)
    stock_extended.ta.zlma(append=True) # Zero Lag Moving Average (ZLMA)
    
    # Statistics (11)
    stock_extended.ta.entropy(append=True) # Entropy
    stock_extended.ta.kurtosis(append=True) # Kurtosis
    stock_extended.ta.mad(append=True) # Mean Absolute Deviation (MAD)
    stock_extended.ta.median(append=True) # Median
    stock_extended.ta.quantile(append=True) # Quantile
    stock_extended.ta.skew(append=True) # Skew
    stock_extended.ta.stdev(append=True) # Standard Deviation (STDEV)
    stock_extended.ta.tos_stdevall(append=True) # Think or Swim Standard Deviation All (TOS_STDEVALL)
    stock_extended.ta.variance(append=True) # Variance
    stock_extended.ta.zscore(append=True) # Z Score
    
    # Performance (3)
    # stock_extended.ta.drawdown(append=True) # Draw Down (DRAWDOWN)
    stock_extended.ta.log_return(append=True) # Log Return (LOG_RETURN)
    stock_extended.ta.percent_return(append=True) # Percent Return (PERCENT_RETURN)
    
    
    smis_extended = {}
    for name, df in smis_dict.items():
        df.ta.alma(append=True)
        df.ta.dpo(append=True)
        df.ta.mom(append=True)
        df.ta.fwma(append=True)
        df.columns = [f'{name}_{col}' for col in df.columns]
        smis_extended[name] = df
    
    trend_extended = {}
    for trend, df in trends.items():
        extended_df = pd.DataFrame(df)
        for column in df.columns:
            alma = df.ta.alma(close=column).to_frame()
            alma.columns = [f'{column}_{col}' for col in alma.columns]
    
            dpo = df.ta.dpo(close=column).to_frame()
            dpo.columns = [f'{column}_{col}' for col in dpo.columns]
    
            mom = df.ta.mom(close=column).to_frame()
            mom.columns = [f'{column}_{col}' for col in mom.columns]
    
            fwma = df.ta.fwma(close=column).to_frame()
            fwma.columns = [f'{column}_{col}' for col in fwma.columns]
    
            extended_df = pd.concat([extended_df, alma, dpo, mom, fwma], axis=1)

        trend_extended[trend] = extended_df

    return stock_extended, smis_extended, trend_extended


def merge_dataframes(stock_extended, smis_extended, trend_extended):
    '''
    Concatenates stock_extended will relevant dataframes in smis_extended
    and trend_extended.
    
    Inputs
    ----------
    stock_extended: Pandas DataFrame with dates as indices
        Contains stock market data with technical indicators
    smis_extended: Dictionary of DataFrames
        Contains SMIS data with technical indicators
    trend_extended: Dictionary of DataFrames
        Contains internet trend data with technical indicators
            
    Outputs
    ----------
    concatenated_df: Pandas DataFrame with dates as indices
        The concatenation of all relevant data
    '''
    list_smis = list(smis_extended.values())
    list_trend = list(trend_extended.values())
    dataframes = [stock_extended] + list_smis + list_trend
    
    concatenated_df = dataframes[0].copy()

    for df in dataframes[1:]:
        concatenated_df = pd.concat([concatenated_df, df], axis=1)

    concatenated_df = concatenated_df.interpolate(method='time')

    return concatenated_df

def partition_data_for_LSTM(features,
                            seq_len,
                            horizon):
    '''
    Transform the time series into a 3D input for the LSTM
    
    Inputs
    ----------
    features: Pandas DataFrame with dates as indices
        The time series with engineered faetures
    
    seq_len: Positive Integer
        Sequence length of input data
    
    horizon: Positive Integer
        How many days ahead the LSTM should forecast
            
    Outputs
    ----------
    X: A numpy array of dimension (samples, n_timesteps, n_features)
        Data that can be used for training the LSTM or making predictions
    '''
    m = len(features) - seq_len - horizon
    n_features = features.shape[1]
    
    X = np.zeros((m, seq_len, n_features))
    
    for i in range(m):
        X[i, :, :] = features[i:i+seq_len, :]
    
    return X

def get_target(time_series,
               horizon,
               target_column='Close'):
    '''
    Computes the target array of returns based on target_column and horizon.
    For example, Return[i] = Close[i+horizon]/Close[i] - 1
    
    Inputs
    ----------
    time_series: Pandas DataFrame
        The time series with predictive features
    horizon: Positive Integer
        How many days ahead the prediction should be
    target_column: string
        Name of the column used in time_series to calculate the target
            
    Output
    ----------
    y: A 1D numpy array
        The computed targets
    '''
    
    m = len(time_series) - horizon
    
    y = np.zeros((m,)) 
    
    for i in range(m):
        current_value = time_series[target_column].iloc[i]
        next_value = time_series[target_column].iloc[i+horizon]
        y[i] = next_value/current_value - 1
    
    return y

if (__name__  == '__main__'):
    from src.data.extract_dataset import extract_financial_data
    stock_dict, smis_dict, trend_dict = extract_financial_data(data_dir = '../../data', 
                                                               save=False, online=False)
    stock = stock_dict['AAPL']
    from copy import deepcopy
    stock_extended, smis_extended, trend_extended = compute_technical_indicators(stock,
                                                                                 deepcopy(smis_dict),
                                                                                 deepcopy(trend_dict['AAPL']))

    features = merge_dataframes(stock_extended, smis_extended, trend_extended)
    df_features = features.dropna()
    
    horizon = 1
    target_samples_per_day = 1
    
    sampled_features = sample_features(df_features,
                                       target_samples_per_day = target_samples_per_day,
                                       target = 'Close',
                                       bar_type = 'dollar',
                                       imbalance_sign = False,
                                       beta = 10000)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(sampled_features)
    
    seq_len = 30
    X = partition_data_for_LSTM(features_scaled,
                                int(seq_len*target_samples_per_day),
                                int(horizon*target_samples_per_day))
