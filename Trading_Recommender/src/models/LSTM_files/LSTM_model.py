from tensorflow.keras.layers import LSTM, Dense, Concatenate, Identity
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanSquaredError


def LSTM_model(args,
               training = True):
    '''
    Fits the forecaster's model.
    
    Inputs
    ----------
    args:     Defaultdict(int), the updated arguments for the model definition
    training: Boolean, whether the model is defined for training (True)
              or for inference (False)
        
    Outputs
    ----------
    model:    The defined and compiled keras LSTM model
    '''
    ## 1) Define the LSTM layer
    if training:
        stateful = args['stateful_training']
    else:
        stateful = args['stateful_inference']
    
    LSTM_layer = LSTM(args['n_a'],
                      return_state = stateful, 
                      return_sequences = False,
                      stateful = stateful,
                      dropout = args['dropout'],
                      recurrent_dropout = args['drop_out'],
                      name = 'LSTM_layer')
    
    ## 2) Define the Concatenate layer to concatenate the hidden states
    if stateful:
        Concatenate_layer = Concatenate(axis=1,
                                        name='Concatenate')
    else:
        Concatenate_layer = Identity(name='Identity')
    
    
    ## 3) Define the second dense layer
    n_out = 1
    dense_layer = Dense(n_out,
                        activation = 'linear',
                        name = 'dense_layer')
    
    # 4) Create and compile the model
    model = Sequential(
        [
            LSTM_layer,
            Concatenate_layer,
            dense_layer
        ]
    )
    
    from src.models.LSTM_files import losses
    if hasattr(losses, args['loss']):
        loss_function = getattr(losses, args['loss']) # Custom loss function
    else:
        loss_function = args['loss'] # Standard loss function
        
    model.compile(loss = loss_function,
                  optimizer = Adam(learning_rate = args['learning_rate'],
                                   beta_1 = 0.9, # Could become a key of args
                                   beta_2 = 0.999, # Could become a key of args
                                   amsgrad = False),
                  metrics = ['mean_absolute_error'])
    
    return model