"""
Helper functions to build up the networks and make it looks neat
"""
import tensorflow as tf

def leaky_relu(inputs, alpha = 0.01):
    return 0.5 * (1 + alpha) * inputs + 0.5 * (1-alpha) * tf.abs(inputs)

def activation_fun(inputs, activation_term):
    """ control which activation functions to use after the convolution layer
    arguments: 
    inputs: an input tensor
    activation_term: the activation function to use
    
    return: an output tensor
    """
    
    if activation_term == 'relu':
        outputs = tf.nn.relu(inputs)
        
    elif activation_term == 'elu':
        outputs = tf.nn.elu(inputs)
        
    elif activation_term == 'crelu':
        outputs = tf.nn.crelu(features=inputs)
        
    elif activation_term == 'softplus':
        outputs = tf.nn.softplus(inputs)
        
    elif activation_term == 'leaky_relu':
        outputs = leaky_relu(inputs)
    
    return outputs

def optimizer_fun(optimizer_term, min_term, learning_rate=1e-3):
    """ control which optimizer to use 
    arguments: 
    optimizer_term: the optimizer to use
    min_term: the viriable to minimize
    
    return: train_op: a well-defined optimizer
    """
    
    if optimizer_term == 'rmsprop':
        train_op = tf.compat.v1.train.RMSPropOptimizer(learning_rate, 
    ).minimize(min_term)
        
    elif optimizer_term == 'adam':
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate, 
    ).minimize(min_term)
    
    elif optimizer_term == 'adagrad':
        train_op = tf.compat.v1.train.AdagradOptimizer(learning_rate, 
    ).minimize(min_term)
        
    elif optimizer_term == 'adadelta':
        train_op = tf.compat.v1.train.AdadeltaOptimizer(learning_rate, 
    ).minimize(min_term)
    
    elif optimizer_term == 'momentum':
        train_op = tf.compat.v1.train.MomentumOptimizer(learning_rate, 0.9
    ).minimize(min_term)
    
    return train_op