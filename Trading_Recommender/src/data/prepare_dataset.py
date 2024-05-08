import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def process_data(time_series,
                 order,
                 direction='pack',
                 scaler=None,
                 first_values=None):
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
        time_series, first_values = difference(time_series, order)
        time_series, scaler = scale(time_series)
        return time_series, scaler, first_values
    if direction == 'unpack':
        time_series = unscale(time_series, scaler)
        time_series = integrate(time_series, first_values, order)
        return time_series

def difference(time_series,
               order):
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
    first_values = []
    for _ in range(order): 
        first_values.append(time_series.iloc[0])
        time_series = time_series.diff(axis=0).ffill().bfill()
    return time_series, first_values[::-1]

def integrate(time_series,
              first_values,
              order):
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
    for _ in range(order):
        first_value = first_values[_]
        time_series.iloc[0] = first_value
        time_series = time_series.cumsum(axis=0)
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
    data = extract_financial_data()[['y']]
    time_series = data[list(data.keys())[0]]
    
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
    order = 3
    debug_series, first_values = difference(time_series, order)
    integrated_series = integrate(debug_series, first_values, order)
    wrong_values, n_errors = compare(integrated_series, time_series)
    print(f'There are {n_errors} errors in the integrated series')

    # Testing process_data
    order = 1
    processed, scaler, first_values = process_data(time_series,
                                                   order=order,
                                                   direction='pack',
                                                   scaler=None)
    unprocessed = process_data(processed,
                               order=order,
                               direction='unpack',
                               scaler=scaler,
                               first_values=first_values)
    wrong_values, n_errors = compare(unprocessed, time_series)
    print(f'There are {n_errors} errors in the unprocessed series')
