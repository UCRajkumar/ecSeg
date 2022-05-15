import tensorflow as tf
import numpy as np

# originally from https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/
def generate_anchors(anchors_reference, anchor_stride, feature_map_shape):
    """Generate anchor for an image.
    Using the feature map, the output of the pretrained network for an
    image, and the anchor_reference generated using the anchor config
    values. We generate a list of anchors.
    Anchors are just fixed bounding boxes of different ratios and sizes
    that are uniformly generated throught the image.
    Args:
        feature_map_shape: Shape of the convolutional feature map used as
            input for the RPN. Should be (batch, height, width, depth).
    Returns:
        all_anchors: A flattened Tensor with all the anchors of shape
            `(num_anchors_per_points * feature_width * feature_height, 4)`
            using the (x1, y1, x2, y2) convention.
    """
    with tf.compat.v1.variable_scope('generate_anchors'):
        grid_width = feature_map_shape[1]  # width
        grid_height = feature_map_shape[0]  # height
        shift_x = tf.range(grid_width) * anchor_stride
        shift_y = tf.range(grid_height) * anchor_stride
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        shifts = tf.stack(
            [shift_x, shift_y, shift_x, shift_y],
            axis=0
        )

        shifts = tf.transpose(a=shifts)
        # Shifts now is a (H x W, 4) Tensor

        # Expand dims to use broadcasting sum.
        all_anchors = (
            tf.expand_dims(anchors_reference, axis=0) +
            tf.expand_dims(shifts, axis=1)
        )

        # Flatten
        all_anchors = tf.reshape(
            all_anchors, (-1, 4)
        )
    return tf.cast(all_anchors, tf.float32)