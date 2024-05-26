import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tsfracdiff import FractionalDifferentiator

# For univariate time series only

def process_data(time_series,
                 direction='pack',
                 scaler=None,
                 fracDiff=None,
                 lagData=None):
    '''
    Process the time series according to the following sequence:
        Raw Time Series --> Differenced Time Series --> Scaled Time Series
        // A model is trained on it and produces a forecast (Time Series) //
        Scaled Time Series -->> Differenced Time Series --> Raw Time Series
        
    Input
    ----------
    time_series: Pandas DataFrame, time series to process
    order:       Positive Integer, order of differencing
    direction:   String:
                        difference & scale if 'pack'
                        unscale & integrate if 'unpack'
    scaler:      The scaler used for scaling
    
    Output
    ----------
    time_series: Pandas DataFrame, processed time series
    scaler:      The scaler used for scaling
    '''
    
    if direction == 'pack':
        time_series, fracDiff, lagData = difference(time_series)
        time_series, scaler = scale(time_series)
        return time_series, scaler, fracDiff, lagData
    if direction == 'unpack':
        time_series = unscale(time_series, scaler)
        time_series = integrate(time_series, fracDiff, lagData)
        return time_series

def difference(time_series):
    '''
    Differenciate the time series 'order' times to make it stationary
    
    Input
    ----------
    time_series: An unscaled, non stationary time series
    order:       Positive Integer, order of differencing
    
    Output
    ----------
    time_series: An unscaled, stationary time series
    '''
    fracDiff = FractionalDifferentiator()
    time_series = fracDiff.FitTransform(time_series)
    print(f'Orders of differencing: {fracDiff.orders}')
    lagData = time_series.head(max(fracDiff.numLags))
    return time_series, fracDiff, lagData

def integrate(time_series,
              fracDiff,
              lagData):
    '''
    Integrate the time series 'order' times to make it stationary
    
    Input
    ----------
    time_series: An unscaled differenced time series
    order:       Positive Integer, order of differencing
    
    Output
    ----------
    time_series: The unscaled integrated time series
    '''
    time_series = fracDiff.InverseTransform(time_series, lagData)
    return time_series
        

def scale(time_series):
    '''
    Scale the differenced stationary time series
    
    Input
    ----------
    time_series: An unscaled differenced time series
    
    Output
    ----------
    time_series: The unscaled integrated time series
    scaler:      The scaler that was used to scale the time series
    '''
    scaler = StandardScaler().fit(time_series)
    dates = time_series.index; names = time_series.columns
    scaled_values = scaler.transform(time_series)
    time_series = pd.DataFrame(scaled_values, index=dates, columns=names)
    return time_series, scaler

def unscale(time_series, scaler):
    '''
    Unscale the differenced stationary time series
    
    Input
    ----------
    time_series: An unscaled differenced time series
    scaler:      The scaler that was used to scale the time series
    
    Output
    ----------
    time_series: The unscaled integrated time series
    '''
    unscaled_values = scaler.inverse_transform(time_series)
    dates = time_series.index; names = time_series.columns
    time_series = pd.DataFrame(unscaled_values, index=dates, columns=names)
    return time_series

if (__name__ == '__main__'):
    from extract_dataset import extract_financial_data
    data = extract_financial_data(data_dir='../../data')
    time_series = data[list(data.keys())[0]][['Open']]
    
    def compare(series_a, series_b, tolerance=0.1):
        mask = (series_a.sub(series_b).abs() > tolerance)
        n_errors = len(np.where(mask)[0])
        return series_a[mask], n_errors
    
    # Testing scale and unscale
    debug_series, debug_scaler = scale(time_series)
    unscaled_series = unscale(debug_series, debug_scaler)
    wrong_values, n_errors = compare(unscaled_series, time_series)
    print(f'There are {n_errors} errors in the unscaled series')
    
    # Testing difference and integrate
    debug_series, fracDiff, lagData = difference(time_series)
    integrated_series = integrate(debug_series, fracDiff, lagData)
    wrong_values, n_errors = compare(integrated_series, time_series)
    print(f'There are {n_errors} errors in the integrated series')

    # Testing process_data
    processed, scaler, fracDiff, lagData = process_data(time_series,
                                                        direction='pack')
    unprocessed = process_data(processed,
                               direction='unpack',
                               scaler=scaler,
                               fracDiff=fracDiff,
                               lagData=lagData)
    wrong_values, n_errors = compare(unprocessed, time_series)
    print(f'There are {n_errors} errors in the unprocessed series')
