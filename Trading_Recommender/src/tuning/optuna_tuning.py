import optuna
from optuna.samplers import TPESampler

from src.forecast.forecaster import forecaster
from src.forecast.recommender import recommender

def optuna_search(n_trials,
                  storage,
                  study_name,
                  args,
                  na_range = (4, 64),
                  lr_range = (0.0001, 0.01),
                  seq_len_range = (60, 180),
                  dropout_range = (0, 0.6)):
    '''
    Performs a LSTM hyperparameter optimization using optuna.
    
    The performance metric is the MAE of the recommender's forecasts.
    
    The parameters tested are:
        the number of LSTM units n_a, 
        the LSTM learning rate learning_rate,
        the LSTM input sequence length seq_len.
        
    Inputs
    ----------
    n_trials: Integer
    How many sets of parameters will be tested
    
    storage: JournalFile
    File where the study will be saved
    
    args: tuple
    Contains arguments necessary to run a trial

    n_a_list: list
    List of possible n_a values to be tested

    lr_range: tuple
    Range of possible learning_rate values to be tested

    seq_len_range: tuple
    Range of possible seq_len values to be tested

    Outputs
    ----------
    n_a: Integer, best number of LSTM units
    learning_rate: Float, best learning rate
    seq_len: Integer, best input sequence length
    '''   
    
    # Unpack the arguments
    model_args, data_args, temporal_args, training_args, trading_args = args
    
    # Define a hyperparameter sampler method
    sampler = TPESampler(seed = 0)
    
    # Create an optuna study
    study = optuna.create_study(storage = storage,
                                study_name = study_name,
                                load_if_exists = True,
                                direction = "minimize",
                                sampler = sampler)
    
    def objective(trial):
        '''
        The optuna objective to minimize.
        
        Inputs
        ----------
        An optuna trial
        
        Outputs
        ----------
        This trial's performance metric
        '''
        # Suggest values for the parameters to optimize
        n_a = trial.suggest_int("n_a", na_range[0], na_range[1])
        learning_rate = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
        seq_len = trial.suggest_int("seq_len", seq_len_range[0], seq_len_range[1])
        dropout = trial.suggest_float("dropout", dropout_range[0], dropout_range[1])
        
        # Assign the values
        model_args['n_a'] = n_a
        model_args['learning_rate'] = learning_rate
        model_args['seq_len'] = seq_len
        model_args['dropout'] = dropout
            
        # Create a forecaster object
        predictor_name = 'LSTM'
        clairvoyant = forecaster(predictor_name, 
                                 model_args)
        
        # Create a recommender object
        recommend = recommender(oracle = clairvoyant,
                                trading_args = trading_args)
        
        
        # Simulate forecasting and recommendations
        recommend(data_args,
                  temporal_args,
                  training_args) # performs the recommendation
        
        # Get the performance metric
        MSE = recommend.MSE
        
        return MSE
    
    # Launch the optuna study
    study.optimize(objective, 
                   n_trials = n_trials)
    
    # Print the best trial, its performance metric and its parameters
    print("\nNumber of finished trials: ", len(study.trials))
    print("\nBest trial:")
    best_trial = study.best_trial
    
    print("  MSE: ", best_trial.value)
    
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Get the best parameters
    n_a = best_trial.params['n_a']
    learning_rate = best_trial.params['learning_rate']
    seq_len = best_trial.params['seq_len']
    
    return n_a, learning_rate, seq_len
