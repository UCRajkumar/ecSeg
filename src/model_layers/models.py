import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def UNET(nb_classes, inputs):
    """Compile a UNET model.

    Args:
        nb_classes: the number of classes to predict
        inputs: the input tensor

    Returns:
        an output tensor, with 'nb_classes' of featuremaps

    """

    padding = 'same'
    # Conv block 1
    outputs = tf.compat.v1.layers.conv2d(inputs, 64, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv1-1', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    outputs = tf.compat.v1.layers.conv2d(outputs, 64, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv1-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    # Make a copy of conv1 output tensor 
    conv1_output = outputs
    
    # Down-sample 1
    outputs = tf.compat.v1.layers.max_pooling2d(outputs,pool_size = 2,strides = 2,padding=padding)
    
    # Conv block 2
    outputs = tf.compat.v1.layers.conv2d(outputs, 128, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv2-1', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.compat.v1.layers.conv2d(outputs, 128, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv2-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    # Make a copy of conv2 output tensor 
    conv2_output = outputs
    
    # Down-sample 2
    outputs = tf.compat.v1.layers.max_pooling2d(outputs,pool_size = 2,strides = 2,padding=padding)
    
    # Conv block 3
    outputs = tf.compat.v1.layers.conv2d(outputs, 256, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv3-1', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.compat.v1.layers.conv2d(outputs, 256, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv3-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    # Make a copy of conv3 output tensor 
    conv3_output = outputs
    
    # Down-sample 3
    outputs = tf.compat.v1.layers.max_pooling2d(outputs,pool_size = 2,strides = 2,padding=padding)
    
    # Conv block 4
    outputs = tf.compat.v1.layers.conv2d(outputs, 512, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv4-1', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.compat.v1.layers.conv2d(outputs, 512, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv4-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    # Make a copy of conv4 output tensor 
    conv4_output = outputs
    
    # Down-sample 4
    outputs = tf.compat.v1.layers.max_pooling2d(outputs,pool_size = 2,strides = 2,padding=padding)
    
    # Get extracted feature for RPN
    rpn_feature = outputs
    
    
    # Conv block 5
    outputs = tf.compat.v1.layers.conv2d(outputs, 1024, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv5-1', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.compat.v1.layers.conv2d(outputs, 1024, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv5-2', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    
    # Up-sample(Conv_transpose) 4
    outputs = tf.compat.v1.layers.conv2d_transpose(outputs, 512, 3, strides=(2, 2),
 padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # changing the former line to the line below will connect the 4th layer on both ends, 
    # and form the classic U-Net architecture, but in our experiments we found this did 
    # not gain a better performance, also thanks for Meryem Uzun-Per for pointing this out
    
    # outputs = tf.concat([conv4_output, outputs], 3)

    
    # Conv block 4'
    outputs = tf.compat.v1.layers.conv2d(outputs, 512, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv4-3', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.compat.v1.layers.conv2d(outputs, 512, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv4-4', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # Up-sample(Conv_transpose) 3
    outputs = tf.compat.v1.layers.conv2d_transpose(outputs, 256, 3, strides=(2, 2),
 padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), use_bias=True)
    outputs = tf.concat([conv3_output, outputs], 3)
    
    # Conv block 3'
    outputs = tf.compat.v1.layers.conv2d(outputs, 256, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv3-3', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.compat.v1.layers.conv2d(outputs, 256, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv3-4', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # Up-sample(Conv_transpose) 2
    outputs = tf.compat.v1.layers.conv2d_transpose(outputs, 128, 3, strides=(2, 2),
 padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), use_bias=True)
    outputs = tf.concat([conv2_output, outputs], 3)
    
    # Conv block 2'
    outputs = tf.compat.v1.layers.conv2d(outputs, 128, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv2-3', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.compat.v1.layers.conv2d(outputs, 128, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv2-4', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # Up-sample(Conv_transpose) 1
    outputs = tf.compat.v1.layers.conv2d_transpose(outputs, 64, 3, strides=(2, 2),
 padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), use_bias=True)
    outputs = tf.concat([conv1_output, outputs], 3)
    
    # Conv block 2'
    outputs = tf.compat.v1.layers.conv2d(outputs, 64, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv1-3', use_bias=True)
    outputs = tf.nn.relu(outputs)

    outputs = tf.compat.v1.layers.conv2d(outputs, 64, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='conv1-4', use_bias=True)
    outputs = tf.nn.relu(outputs)
    
    # only output 2 featuremaps at the end
    outputs = tf.compat.v1.layers.conv2d(outputs, nb_classes, 3, padding=padding, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='final', use_bias=False)
    
    return outputs, rpn_feature
