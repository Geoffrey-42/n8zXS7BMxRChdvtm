import tensorflow as tf
from tensorflow.keras import backend as K

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    quadratic_term = tf.minimum(tf.abs(error), delta)
    linear_term = tf.abs(error) - quadratic_term
    return 0.5 * tf.square(quadratic_term) + delta * linear_term

def quantile_loss(q, y_true, y_pred):
    error = y_true - y_pred
    return K.mean(K.maximum(q * error, (q - 1) * error), axis=-1)

def logcosh_loss(y_true, y_pred):
    log_pred = K.log(y_pred + K.epsilon())
    log_true = K.log(y_true + K.epsilon())
    loss = K.mean(K.log(K.cosh(log_pred - log_true) + K.epsilon()), axis=-1)
    return loss

def log_loss(y_true, y_pred):
    error = y_true - y_pred
    return K.mean(K.log(1 + K.abs(error)), axis=-1)