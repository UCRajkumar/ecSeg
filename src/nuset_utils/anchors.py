import numpy as np
import tensorflow as tf

# originally from https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/

def generate_anchors_reference(base_size, aspect_ratios, scales):
    """Generate base anchor to be used as reference of generating all anchors.

    Anchors vary only in width and height. Using the base_size and the
    different ratios we can calculate the wanted widths and heights.

    Scales apply to area of object.

    Args:
        base_size (int): Base size of the base anchor (square).
        aspect_ratios: Ratios to use to generate different anchors. The ratio
            is the value of height / width.
        scales: Scaling ratios applied to area.

    Returns:
        anchors: Numpy array with shape (total_aspect_ratios * total_scales, 4)
            with the corner points of the reference base anchors using the
            convention (x_min, y_min, x_max, y_max).
    """
    scales_grid, aspect_ratios_grid = tf.meshgrid(scales, aspect_ratios)
    base_scales = tf.reshape(scales_grid, [-1])
    base_aspect_ratios = tf.reshape(aspect_ratios_grid, [-1])

    aspect_ratio_sqrts = tf.sqrt(base_aspect_ratios)
    heights = base_scales * aspect_ratio_sqrts * base_size
    widths = base_scales / aspect_ratio_sqrts * base_size

    # Center point has the same X, Y value.
    center_xy = 0

    # Create anchor reference.
    anchors = tf.stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2,
    ], axis = -1)

    real_heights = tf.cast(anchors[:, 3] - anchors[:, 1], tf.int32)
    real_widths = tf.cast(anchors[:, 2] - anchors[:, 0], tf.int32)

    """
    if (real_widths == 0).any() or (real_heights == 0).any():
        raise ValueError(
            'base_size {} is too small for aspect_ratios and scales.'.format(
                base_size
            )
        )
    """
    return anchors
