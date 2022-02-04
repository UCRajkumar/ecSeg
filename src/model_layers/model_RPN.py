import tensorflow as tf
import numpy as np


def RPN(feat_map, num_refanchors):
    
    """Compile a region proposal network
    With reference to https://github.com/endernewton/tf-faster-rcnn/blob/master/lib/nets/network.py
    and https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/rpn.py
    Args:
        feat_map (tensor): the parameters of the genome
        num_refanchors(num): number of reference anchors placed on the feature map

    Returns:
        a compiled network. 

    """
    initializer = tf.compat.v1.random_normal_initializer(mean=0.0, stddev=0.01)
    initializer_bbox = tf.compat.v1.random_normal_initializer(mean=0.0, stddev=0.001)
    prediction_dict = {}
    
    # pass it once to get the feature map ready to split into cls and reg
    rpn = tf.compat.v1.layers.conv2d(feat_map, 512, [3, 3], kernel_initializer=initializer,
                           padding='same', name="rpn_conv/3x3")
    # This is the score to determine an anchor is foreground vs background
    rpn_cls_score = tf.compat.v1.layers.conv2d(rpn, num_refanchors * 2, [1, 1],
                                kernel_initializer=initializer,
                                padding='valid', name='rpn_cls_score')
    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = tf.reshape(rpn_cls_score, [-1, 2])
    # Now the shape of rpn_cls_score_reshape is (H * W * num_anchors, 4)
    rpn_cls_prob = tf.nn.softmax(rpn_cls_score_reshape)
    
    # rpn_bbox_pred has shape (?, H, W, num_anchors * 4)
    rpn_bbox_pred = tf.compat.v1.layers.conv2d(rpn, num_refanchors * 4, [1, 1],
                                kernel_initializer=initializer_bbox,
                                     padding='VALID', name='rpn_bbox_pred')
    # change it so that the pred has 4 as its channel size (4 coordinates)
    rpn_bbox_pred_reshape = tf.reshape(rpn_bbox_pred, [-1, 4])
    
    # Add the output to the dictionary
    prediction_dict['rpn_cls_prob'] = rpn_cls_prob
    prediction_dict['rpn_cls_score'] = rpn_cls_score_reshape
    prediction_dict['rpn_bbox_pred'] = rpn_bbox_pred_reshape
     
    return prediction_dict