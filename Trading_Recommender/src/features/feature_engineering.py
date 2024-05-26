import warnings
from itertools import accumulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

from src.features.extend_dataset import get_target
from src.data.extract_dataset import extract_financial_data
from src.features.extend_dataset import compute_technical_indicators, merge_dataframes
from src.features.feature_selection import select_features, evaluate_features
from copy import deepcopy

def engineer_features(features_selected,
                      data_dir = '../../data',
                      variance_threshold = 0.9):
    '''
    Engineer orthogonal features using PCA from sktime.
        
    Input
    ----------
    features_selected: Time series in sktime compatible data format
        The time series extended with selected technical features
        
    data_dir: String
        Path to the local folder containing the raw data
    
    variance_threshold: Float between 0 and 1
        Minimum amount of total variance of the selected Principal Components 
    
    Output
    ----------
    features_engineered: Time series in sktime compatible data format
        The time series with engineered features
        
    transformer: PCATransformer instance from sktime
        The fitted PCA
    '''
    features_scaled, _ = scale_dataframe(features_selected)
    transformer = PCA()
    features_engineered = transformer.fit_transform(features_scaled)
    
    features_engineered = pd.DataFrame(features_engineered)
    features_engineered.index = features_selected.index
    features_engineered.columns = [f'PC{i+1}' for i in range(features_engineered.shape[1])]
    
    explained_variance = transformer.explained_variance_/sum(transformer.explained_variance_)
    n_transformed = next((i for i, total in enumerate(accumulate(explained_variance)) if total > variance_threshold))
    print(f'\nOnly {n_transformed+1} features are extracted from the PCA with a total relative variance of {sum(explained_variance[:n_transformed+1])}/1\n')
    features_engineered = features_engineered.iloc[:, :n_transformed+1]
    
    return features_engineered, transformer

def scale_dataframe(df):
    'Returns a copy of the df dataframe, but scaled by mean and variance'
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.copy())
    scaled = pd.DataFrame(scaled, columns = df.columns, index = df.index)
    return scaled, scaler

def unscale_dataframe(df, scaler):
    'Returns the unscaled version of the df dataframe which was scaled by scaler'
    unscaled = scaler.inverse_transform(df.copy())
    unscaled = pd.DataFrame(unscaled, columns = df.columns, index = df.index)
    return unscaled

def get_final_dataframe(features_engineered,
                        target):
    '''
    Concatenates a scaled version of the target array
    to the engineered features dataframe.
    
    Input
    ----------
    features_engineered: Pandas DataFrame with dates as indices
        The dataframe with engineered features
    
    target: 1D numpy array
        The unscaled targets
    
    Output
    ----------
    final_df: Pandas DataFrame with dates as indices
        The dataframe with scaled engineered features and the scaled target
    
    scaler: sklearn scaler
        Scaler used for the final dataframe
    '''
    target_df = pd.DataFrame(data = target, columns = ['Return'], index = features_engineered.index[:len(target)])
    
    scaled_features, f_scaler = scale_dataframe(features_engineered)
    scaled_target, t_scaler = scale_dataframe(target_df)
    assert np.isclose(f_scaler.inverse_transform(scaled_features), features_engineered).all()
    assert np.isclose(t_scaler.inverse_transform(scaled_target), target_df).all()
    
    scalers = (f_scaler, t_scaler)
    final_df = pd.concat([scaled_features.iloc[:len(target)], scaled_target], axis = 1)
    
    return final_df, scalers

def get_engineered_features(stock_name = 'AAPL',
                            data_dir = '../../data',
                            horizon = 14,
                            variance_threshold = 0.9):
    '''
    Executes the entire process of extracting the raw data and engineering 
    relevant features out of it.
    
    Input
    ----------
    stock_name: String
        Alias of selected stock
        
    data_dir: String
        Path to the local folder containing the raw data
    
    horizon: Positive Integer
        Horizon of desired forecast in days. 
        The return target is calcuted with: Close[i+horizon]/Close[i] - 1
    
    variance_threshold: Float between 0 and 1
        Minimum amount of total variance of the selected Principal Components
    
    Output
    ----------
    features_engineered: Time series in sktime compatible data format
        The time series with engineered features
        
    scalers: Tuple of StandardScaler instances from scikit-learn
        Contains scaler used on engineered features (post PCA) and target
        
    transformer: PCATransformer instance from sktime
        The fitted PCA used on the selected features
        
    feature_names: List
        List of selected features names
    
    close: Pandas DataFrame
        Contains the original Close stock historical data for the stock selected
    '''
    stock_dict, smis_dict, trend_dict = extract_financial_data(data_dir = data_dir, 
                                                               save=False, online=False)
    stock_name = 'AAPL'
    stock = stock_dict[stock_name]
    close = stock[['Close']]

    stock_extended, smis_extended, trend_extended = compute_technical_indicators(stock,
                                                                                 deepcopy(smis_dict),
                                                                                 deepcopy(trend_dict[stock_name]))

    extended_df_withna = merge_dataframes(stock_extended, smis_extended, trend_extended)
    extended_df = extended_df_withna.dropna()
    horizon = 14
    target = get_target(extended_df,
                        horizon)
    y = (target>0).astype(int)
    
    
    regressor = GradientBoostingRegressor(max_depth=1)
    features_selected, feature_importances = select_features(extended_df,
                                                             horizon = horizon,
                                                             n_columns = None,
                                                             regressor = regressor,
                                                             importance_threshold = 0.99)
    print(f'Using {regressor = }, only {features_selected.shape[1]} features are selected')
    print(f'The selected features with {horizon = } are: \n{features_selected.columns}')
    print(f'With a cumulated relative importance of {sum(map(lambda item: item[1], feature_importances))}')
    
    untransformed_result = evaluate_features(scale_dataframe(features_selected)[0], y)
    untransformed_accuracies = (untransformed_result['HIVECOTEV2_accuracy'],
                                untransformed_result['Rocket_accuracy'])
    print(f'The HIVECOTEV2 and Rocket classifiers respective accuracies using the {features_selected.shape[1]} untransformed features are: {untransformed_accuracies}')
    
    features_engineered, transformer = engineer_features(features_selected,
                                                         data_dir = data_dir,
                                                         variance_threshold = variance_threshold)
    
    transformed_result = evaluate_features(features_engineered, y)
    transformed_accuracies = (transformed_result['HIVECOTEV2_accuracy'],
                              transformed_result['Rocket_accuracy'])
    print(f'The HIVECOTEV2 and Rocket classifiers respective accuracies using the {features_engineered.shape[1]} transformed features are: {transformed_accuracies}')
    
    final_df, scalers = get_final_dataframe(features_engineered,
                                            target)
    
    feature_names = list(features_selected.columns)
    
    return final_df, scalers, transformer, feature_names, close
    

if (__name__ == '__main__'):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    final_df, scalers, transformer, features_names, close = get_engineered_features()
    
    covariance = transformer.get_covariance()
    explained_variance = transformer.explained_variance_/sum(transformer.explained_variance_)
    components = transformer.components_[:final_df.shape[1]-1, :]
    contributions = np.abs(components)/np.sum(np.abs(components), axis = 1, keepdims=True)
    contributions_df = pd.DataFrame(contributions, columns=features_names, index=[f'PC {i+1} (Variance {explained_variance[i]:.4f})' for i in range(components.shape[0])])
    contributing_features = [sorted([(contributions_df.columns[i], value) for i, value in enumerate(contributions_df.iloc[j])], key = lambda x:x[1], reverse=True) for j in range(contributions_df.shape[0])]
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    for i, component in enumerate(contributing_features[:2]):
        features = [x[0] for x in component]
        values = [x[1] for x in component]
    
        axs[i].bar(features, values)
        axs[i].tick_params(axis='x', rotation=90)
        axs[i].tick_params(axis='x', labelsize=8)
        axs[i].set_title(f'Principal Component {i+1} (Variance {explained_variance[i]:.4f})')
        axs[i].set_xlabel('Features used')
        axs[i].set_ylabel('Contribution of the feature')
    
    plt.tight_layout(pad=3)
    plt.show()