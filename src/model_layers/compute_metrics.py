import tensorflow as tf

def compute_metrics(pred_masks, labels, batch_size = 1):
    
    all_metrics = {}
    all_metrics_op = {}
    
    all_metrics['global'] = {}

    all_metrics_op['global'] = {}

    
    # format the argument for the metrics
    labels_reshape = tf.cast(tf.reshape(labels,[batch_size, -1]), tf.int32)
    pred_masks_reshape = tf.cast(tf.reshape(pred_masks,[batch_size, -1]), tf.int32)
    
    # Global metrics
    all_metrics['global']['pixel_accuracy'], all_metrics_op['global']['pixel_accuracy'] = tf.compat.v1.metrics.accuracy(labels_reshape, pred_masks_reshape, weights = None)
    
    all_metrics['global']['mean_IU'], all_metrics_op['global']['mean_IU'] = tf.compat.v1.metrics.mean_iou(labels_reshape, pred_masks_reshape, num_classes = 2, weights = None)
    
    all_metrics['global']['rmse'], all_metrics_op['global']['rmse'] = tf.compat.v1.metrics.root_mean_squared_error(labels_reshape, pred_masks_reshape, weights = None)
    
    all_metrics['global']['precision'], all_metrics_op['global']['precision'] = tf.compat.v1.metrics.precision(labels_reshape, pred_masks_reshape, weights=None)
    
    all_metrics['global']['recall'], all_metrics_op['global']['recall'] = tf.compat.v1.metrics.recall(labels_reshape, pred_masks_reshape, weights=None)
    
    
    
    return all_metrics, all_metrics_op