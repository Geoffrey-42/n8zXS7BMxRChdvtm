from src.features.extend_dataset import get_target

import pandas as pd
from itertools import accumulate
import warnings

from sktime.transformations.series.feature_selection import FeatureSelection
from sklearn.preprocessing import StandardScaler
# from sktime.transformations.compose import ColumnwiseTransformer

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
# from sktime.classification.deep_learning import InceptionTimeClassifier
# from sktime.classification.interval_based import CanonicalIntervalForest

def select_features(extended_df,
                    horizon = 1,
                    n_columns = None,
                    regressor = None,
                    importance_threshold = 0.99):
    '''
    Select relevant features from extended_df based on a provided regressor
        
    Input
    ----------
    extended_df: Time series in sktime compatible data format
        The time series extended with many technical features
        
    horizon: Positive Integer
        How many days ahead the features should help to forecast for
        
    n_columns: Positive integer
        Max amount of features to select (automatic if None)
        
    regressor: sktime time series regressor
        Regressor used to estimate feature importance
        
    importance_threshold: Float between 0 and 1
        Minimum amount of total relative importance of the selected features
    
    Output
    ----------
    features_selected: Time series in sktime compatible data format
        The time series with selected features
        
    feature_importances: List of tuples
        Ordered list of selected features with their relative importance
    '''
    target = get_target(extended_df,
                        horizon)
    y = (target>0).astype(int)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(extended_df)
    
    train_index = len(extended_df[extended_df.index<pd.to_datetime('2023-01-01')])
    X, y = features_scaled[:train_index, :], y[:train_index]
    
    selector = FeatureSelection(method="feature-importances",
                                regressor = regressor,
                                n_columns = n_columns)
    selector.fit(X, y)
    
    feature_importances = sorted([(extended_df.columns[i], selector.feature_importances_[i]) for i in selector.columns_], key=lambda item:item[1], reverse=True)
    n_features = next((i for i, total in enumerate(accumulate(map(lambda item: item[1], feature_importances))) if total >= importance_threshold), len(feature_importances)-1)
    
    selected_features = extended_df.columns[selector.columns_[:n_features]]
    features_selected = extended_df[selected_features]
    
    return features_selected, feature_importances

def evaluate_features(features_selected,
                      y):
    '''
    Evaluate the selected features in features_selected.
    State-of-the-art time series classifiers are used to evaluate the selected 
    features on an validation set.
        
    Input
    ----------
    features_selected: Time series in sktime compatible data format
        The time series with selected features
    y: List
        Label to predict using the selected features
    
    Output
    ----------
    result: Dictionary
        The evaluation of the selected features
    '''
    # 1) Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_selected)
    # 2) Form train and validation sets
    train_index = len(features_selected[features_selected.index<pd.to_datetime('2022-01-01')])
    val_index = len(features_selected[features_selected.index<pd.to_datetime('2023-01-01')])
    X_train, y_train = features_scaled[:train_index, :], y[:train_index]
    X_val, y_val = features_scaled[train_index:val_index, :], y[train_index:val_index]
    
    # 3) Define the state-of-the-art time series classifiers to be used
    classifiers = [('Rocket', RocketClassifier(num_kernels=2000)),
                   ('HIVECOTEV2', HIVECOTEV2(time_limit_in_minutes=0.4))]
    # ('ITC', InceptionTimeClassifier())
    # ('CIF', CanonicalIntervalForest())
    
    # 4) Evaluate the selected features using the above-defined classifiers
    result = {}
    for classifier in classifiers:
        name = classifier[0]
        cf = classifier[1]
        cf.fit(X_train, y_train)
        y_pred = cf.predict(X_val)
        
        cf_matrix = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cf_matrix.ravel()
        specificity = tn/(tn+fp)
        
        result[f'{name}_accuracy'] =  accuracy_score(y_val, y_pred)
        result[f'{name}_precision'] = precision_score(y_val, y_pred)
        result[f'{name}_recall'] = recall_score(y_val, y_pred)
        result[f'{name}_specificity'] =  specificity
        result[f'{name}_f1_score'] =  f1_score(y_val, y_pred)
        result[f'{name}_confusion_matrix'] =  cf_matrix
    
    return result
    

def select_and_evaluate_features(extended_df,
                                 select_horizon = 14,
                                 eval_horizon = 14,
                                 max_depth_range = range(1, 8),
                                 importance_threshold = 0.99):
    '''
    Select and evaluate relevant features from extended_df.
    A range of GradientBoostingRegressor are tested with different max depths.
    State-of-the-art time series classifiers are used to evaluate the selected 
    features on an validation set.
    The results are returned for each regressor used for selection.
        
    Input
    ----------
    extended_df: Time series in sktime compatible data format
        The time series extended with many technical features
        
    select_horizon: Positive Integer
        How many days ahead the features should help to forecast for
        Parameter used for feature selection
        
    eval_horizon: Positive Integer
        How many days ahead the features should help to forecast for
        Parameter used for feature evaluation
    
    max_depth_range: range
        Range of the GradientBoostingRegressor max_depth
    
    importance_threshold: Float between 0 and 1
        Minimum amount of total relative importance of the selected features
    
    Output
    ----------
    feature_importance_results: A list of dictionaries
        Contains the results for each regressor used for feature selection
    '''
    feature_importance_results = []
    for max_depth in max_depth_range:
        print(f'\nNow using GradientBoostingRegressor with {max_depth = }...\n')
        # 1) Select the regressor
        regressor = GradientBoostingRegressor(max_depth=max_depth)
        # 2) Select the features
        features_selected, feature_importances = select_features(extended_df.copy(),
                                                                 horizon = select_horizon,
                                                                 n_columns = None,
                                                                 regressor = regressor,
                                                                 importance_threshold = importance_threshold)
        # 3) Selected features characteristics
        result = {'max_depth': max_depth,
                  'feature_importances': feature_importances,
                  'cumulated_importance': sum(map(lambda item: item[1], feature_importances)),
                  'n_features': features_selected.shape[1]}
        # 3) Get the target
        target = get_target(extended_df,
                            eval_horizon)
        y = (target>0).astype(int)
        # 4) Evaluate the features
        result.update(evaluate_features(features_selected, y))
        # 5) Print this iteration's general results
        accuracies = [result['Rocket_accuracy'], 
                      result['HIVECOTEV2_accuracy']]
        # result['ITC_accuracy']
        # result['CIF_accuracy']
        print(f"\nUsing a GradientBoostingRegressor with max_depth = {result['max_depth']} resulted in a classifier accuracy of {max(accuracies)} and {result['n_features']} features of {result['cumulated_importance']} cumulated importance\n")
        # 6) Keep track of this iteration
        feature_importance_results.append(result)
    
    return feature_importance_results


if (__name__ == '__main__'):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from src.data.extract_dataset import extract_financial_data
    stock_dict, smis_dict, trend_dict = extract_financial_data(data_dir = '../../data', 
                                                               save=False, online=False)
    stock = stock_dict['AAPL']
    from src.features.extend_dataset import compute_technical_indicators, merge_dataframes
    from copy import deepcopy
    stock_extended, smis_extended, trend_extended = compute_technical_indicators(stock,
                                                                                 deepcopy(smis_dict),
                                                                                 deepcopy(trend_dict['AAPL']))

    extended_df_withna = merge_dataframes(stock_extended, smis_extended, trend_extended)
    extended_df = extended_df_withna.dropna()
    
    feature_importance_results = select_and_evaluate_features(extended_df,
                                                              select_horizon = 10,
                                                              eval_horizon = 10,
                                                              max_depth_range = range(1, 11))
    # horizon = 14
    # n_columns = None
    # regressor = GradientBoostingRegressor(max_depth=1)
    # features_selected, feature_importances = select_features(extended_df,
    #                                                          horizon = horizon,
    #                                                          n_columns = n_columns,
    #                                                          regressor = regressor)
    
    # print(f'Using {regressor = }')
    # print(f'The selected features with {horizon = } are: \n{features_selected.columns}')
    # print(f'The {len(feature_importances)} selected features importances are:\n{feature_importances}')
    # print(f'With a cumulated importance of {sum(map(lambda item: item[1], feature_importances))}')
    