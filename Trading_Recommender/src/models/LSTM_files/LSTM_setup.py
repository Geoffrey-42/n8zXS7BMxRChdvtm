def LSTM_config(args):
    '''
    Configures the LSTM model arguments.
    This function corrects (if necessary) or completes the user input
    
    Parameters
    ----------
    args: Defaultdict(int), contains user defined arguments for the model definition
    
    Outputs
    ----------
    args: Defaultdict(int), the updated arguments for the model definition
    '''
    ## 0) Configure the arguments that were not given by the user
    
    # n_a: Positive integer, dimensionality of the output space (LSTM units)
    n_a = args['n_a']
    if n_a == 0:
        n_a = 32; args['n_a'] = n_a
    
    # seq_len: Positive Integer, number of timesteps in the time series sequence
    seq_len = args['seq_len']
    if seq_len == 0:
        print("Warning: argument 'seq_len' was not specified in the args defaultdict\n")
        print("The sequence length of the training examples fed into the LSTM will be set to 60 by default\n")
        seq_len = 30; args['seq_len'] = seq_len
    
    # loss: String, name of the loss function used for optimization
    loss = args['loss']
    if loss in {'huber', 'quantile', 'logcosh', 'log'}:
        loss = loss + '_loss'
    loss_set = {'huber_loss', 'quantile_loss', 'logcosh_loss', 'log_loss',
                'mean_squared_error', 'mse', 'mean_absolute_error', 'mae', 
                'mean_absolute_percentage_error', 'mape'}
    if loss not in loss_set:
        print('The loss was either not specified or not supported by this recommender.\n')
        print(f'Supported losses are {loss_set}\n')
        print('Loss will be set by default to the huber loss\n')
        loss = 'huber_loss'
    args['loss'] = loss
    
    # learning_rate: Float, learning rate used for optimization
    learning_rate = args['learning_rate']
    if learning_rate == 0:
        learning_rate = 0.0001; args['learning_rate'] = learning_rate
    
    # dropout: Float, dropout rate in the dropout layer
    dropout = args['dropout']
    if True: # No action is performed
        args['dropout'] = dropout
    
    # stateful_training: Boolean (default: False). 
    stateful_training = bool(args['stateful_training']); args['stateful_training'] = stateful_training
    
    # stateful_inference: Boolean (default: False). 
    stateful_inference = bool(args['stateful_inference']); args['stateful_inference'] = stateful_inference
    
    # The stateful mode can be activated to find dependencies between input sequences:
    # The last state for each sample at index i in a batch will be used 
    # as initial state for the sample of index i in the following batch.
    # If stateful is on during training, batch_size is set to be 1.
    
    # batch_size: Positive Integer, size of input batch fed to the LSTM
    batch_size = args['batch_size']
    if batch_size == 0:
        batch_size = 32
    batch_size = batch_size - (batch_size-1)*int(stateful_training) # 1 if stateful mode on
    args['batch_size'] = batch_size
    
    include_dates = bool(args['include_dates']); args['include_dates'] = include_dates
    n_features = min(args['n_features'], 1+int(include_dates))
    args['n_features'] = n_features # if include_dates=True, dates becomes a feature

    return args