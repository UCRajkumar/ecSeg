import tensorflow as tf

from utils.bbox_overlap import bbox_overlap_tf
from utils.bbox_transform_tf import encode as encode_tf


def RPNTarget(all_anchors, num_anchors, gt_boxes, im_shape):
    """RPNTarget: Get RPN's classification and regression targets.

    RPNTarget is responsible for:
      * calculating the correct values for both classification and regression
        problems.
      * defining which anchors and target values are going to be used for the
        RPN minibatch.

    For calculating the correct values for classification (ie. the question of
    "does this anchor refer to an object?") and returning an objectness score,
    we calculate the intersection over union (IoU) between the anchors boxes
    and the ground truth boxes, and use this to categorize anchors. When the
    intersection between anchors and groundtruth is above a threshold, we can
    mark the anchor as an object or as being foreground. In case of not having
    any intersection or having a low IoU value, then we say that the anchor
    refers to background.

    For calculating the correct values for the regression, the problem of
    transforming the fixed size anchor into a more suitable bounding box (equal
    to the ground truth box) only applies to the anchors that we consider to be
    foreground.

    RPNTarget is also responsible for selecting which of the anchors are going
    to be used for the minibatch. This is a random process with some
    restrictions on the ratio between foreground and background samples.

    For selecting the minibatch, labels are not only set to 0 or 1 (for the
    cases of being background and foreground respectively), but also to -1 for
    the anchors we just want to ignore and not include in the minibatch.

    In summary:
      * 1 is positive
          when GT overlap is >= 0.7 (configurable) or for GT max overlap (one
          anchor)
      * 0 is negative
          when GT overlap is < 0.3 (configurable)
      * -1 is don't care
          useful for subsampling negative labels

    Returns:
        labels: label for each anchor
        bbox_targets: bbox regresion values for each anchor
    """

    allowed_border = 0
    # We set clobber positive to False to make sure that there is always at
    # least one positive anchor per GT box.
    clobber_positives = False
    # We set anchors as positive when the IoU is greater than
    # `positive_overlap`.
    positive_overlap = 0.7
    # We set anchors as negative when the IoU is less than
    # `negative_overlap`.
    negative_overlap = 0.3
    # Fraction of the batch to be foreground labeled anchors.
    foreground_fraction = 0.5
    minibatch_size = 256

    # When choosing random targets use `seed` to replicate behaviour.
    seed = None

    """
    We compare anchors to GT and using the minibatch size and the different
    config settings (clobber, foreground fraction, etc), we end up with
    training targets *only* for the elements we want to use in the batch,
    while everything else is ignored.

    Basically what it does is, first generate the targets for all (valid)
    anchors, and then start subsampling the positive (foreground) and the
    negative ones (background) based on the number of samples of each type
    that we want.

    Args:
        all_anchors:
            A Tensor with all the bounding boxes coords of the anchors.
            Its shape should be (num_anchors, 4).
        gt_boxes:
            A Tensor with the ground truth bounding boxes of the image of
            the batch being processed. Its shape should be (num_gt, 5).
            The last dimension is used for the label.
        im_shape:
            Shape of original image (height, width) in order to define
            anchor targers in respect with gt_boxes.

    Returns:
        Tuple of the tensors of:
            labels: (1, 0, -1) for each anchor.
                Shape (num_anchors, 1)
            bbox_targets: 4d bbox targets as specified by paper.
                Shape (num_anchors, 4)
            max_overlaps: Max IoU overlap with ground truth boxes.
                Shape (num_anchors, 1)
    """
    # Keep only the coordinates of gt_boxes
    gt_boxes = gt_boxes[:, :4]
    all_anchors = all_anchors[:, :4]

    # Only keep anchors inside the image
    (x_min_anchor, y_min_anchor,
     x_max_anchor, y_max_anchor) = tf.unstack(all_anchors, axis=1)

    anchor_filter = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(x_min_anchor, -allowed_border),
            tf.greater_equal(y_min_anchor, -allowed_border)
        ),
        tf.logical_and(
            tf.less(x_max_anchor, im_shape[1] + allowed_border),
            tf.less(y_max_anchor, im_shape[0] + allowed_border)
        )
    )

    # We (force) reshape the filter so that we can use it as a boolean mask
    anchor_filter = tf.reshape(anchor_filter, [-1])
    # Filter anchors.
    anchors = tf.boolean_mask(
        tensor=all_anchors, mask=anchor_filter, name='filter_anchors')

    # Generate array with the labels for all_anchors.
    labels = tf.fill((tf.gather(tf.shape(input=all_anchors), [0])), -1)
    labels = tf.boolean_mask(tensor=labels, mask=anchor_filter, name='filter_labels')

    # Intersection over union (IoU) overlap between the anchors and the
    # ground truth boxes.
    overlaps = bbox_overlap_tf(tf.cast(anchors, dtype=tf.float32), tf.cast(gt_boxes, dtype=tf.float32))

    # Generate array with the IoU value of the closest GT box for each
    # anchor.
    max_overlaps = tf.reduce_max(input_tensor=overlaps, axis=1)
    if not clobber_positives:
        # Assign bg labels first so that positive labels can clobber them.
        # First we get an array with True where IoU is less than
        # negative_overlap
        negative_overlap_nonzero = tf.less(
            max_overlaps, negative_overlap)

        # Finally we set 0 at True indices
        labels = tf.compat.v1.where(
            condition=negative_overlap_nonzero,
            x=tf.zeros(tf.shape(input=labels)), y=tf.cast(labels, dtype=tf.float32)
        )
    # Get the value of the max IoU for the closest anchor for each gt.
    gt_max_overlaps = tf.reduce_max(input_tensor=overlaps, axis=0)

    # Find all the indices that match (at least one, but could be more).
    gt_argmax_overlaps = tf.squeeze(tf.equal(overlaps, gt_max_overlaps))
    gt_argmax_overlaps = tf.compat.v1.where(gt_argmax_overlaps)[:, 0]
    # Eliminate duplicates indices.
    gt_argmax_overlaps, _ = tf.unique(gt_argmax_overlaps)
    # Order the indices for sparse_to_dense compatibility
    gt_argmax_overlaps, _ = tf.nn.top_k(
        gt_argmax_overlaps, k=tf.shape(input=gt_argmax_overlaps)[-1])
    gt_argmax_overlaps = tf.reverse(gt_argmax_overlaps, [0])

    # Foreground label: for each ground-truth, anchor with highest overlap.
    # When the argmax is many items we use all of them (for consistency).
    # We set 1 at gt_argmax_overlaps_cond indices
    gt_argmax_overlaps_cond = tf.compat.v1.sparse_to_dense(
        gt_argmax_overlaps, tf.shape(input=labels, out_type=tf.int64),
        True, default_value=False
    )

    labels = tf.compat.v1.where(
        condition=gt_argmax_overlaps_cond,
        x=tf.ones(tf.shape(input=labels)), y=tf.cast(labels, dtype=tf.float32)
    )

    # Foreground label: above threshold Intersection over Union (IoU)
    # First we get an array with True where IoU is greater or equal than
    # positive_overlap
    positive_overlap_inds = tf.greater_equal(
        max_overlaps, positive_overlap)
    # Finally we set 1 at True indices
    labels = tf.compat.v1.where(
        condition=positive_overlap_inds,
        x=tf.ones(tf.shape(input=labels)), y=labels
    )

    if clobber_positives:
        # Assign background labels last so that negative labels can clobber
        # positives. First we get an array with True where IoU is less than
        # negative_overlap
        negative_overlap_nonzero = tf.less(
            max_overlaps, negative_overlap)
        # Finally we set 0 at True indices
        labels = tf.compat.v1.where(
            condition=negative_overlap_nonzero,
            x=tf.zeros(tf.shape(input=labels)), y=labels
        )

    # Subsample positive labels if we have too many
    def subsample_positive():
        # Shuffle the foreground indices
        disable_fg_inds = tf.random.shuffle(fg_inds, seed=seed)
        # Select the indices that we have to ignore, this is
        # `tf.shape(fg_inds)[0] - num_fg` because we want to get only
        # `num_fg` foreground labels.
        disable_place = (tf.shape(input=fg_inds)[0] - num_fg)
        disable_fg_inds = disable_fg_inds[:disable_place]
        # Order the indices for sparse_to_dense compatibility
        disable_fg_inds, _ = tf.nn.top_k(
            disable_fg_inds, k=tf.shape(input=disable_fg_inds)[-1])
        disable_fg_inds = tf.reverse(disable_fg_inds, [0])
        disable_fg_inds = tf.compat.v1.sparse_to_dense(
            disable_fg_inds, tf.shape(input=labels, out_type=tf.int64),
            True, default_value=False
        )
        # Put -1 to ignore the anchors in the selected indices
        return tf.compat.v1.where(
            condition=tf.squeeze(disable_fg_inds),
            x=tf.cast(tf.fill(tf.shape(input=labels), -1), dtype=tf.float32), y=labels
        )

    num_fg = tf.cast(foreground_fraction * minibatch_size, dtype=tf.int32)
    # Get foreground indices, get True in the indices where we have a one.
    fg_inds = tf.equal(labels, 1)
    # We get only the indices where we have True.
    fg_inds = tf.squeeze(tf.compat.v1.where(fg_inds), axis=1)
    fg_inds_size = tf.size(input=fg_inds)
    # Condition for check if we have too many positive labels.
    subsample_positive_cond = fg_inds_size > num_fg
    # Check the condition and subsample positive labels.
    labels = tf.cond(
        pred=subsample_positive_cond,
        true_fn=subsample_positive, false_fn=lambda: labels
    )

    # Subsample negative labels if we have too many
    def subsample_negative():
        # Shuffle the background indices
        disable_bg_inds = tf.random.shuffle(bg_inds, seed=seed)

        # Select the indices that we have to ignore, this is
        # `tf.shape(bg_inds)[0] - num_bg` because we want to get only
        # `num_bg` background labels.
        disable_place = (tf.shape(input=bg_inds)[0] - num_bg)
        disable_bg_inds = disable_bg_inds[:disable_place]
        # Order the indices for sparse_to_dense compatibility
        disable_bg_inds, _ = tf.nn.top_k(
            disable_bg_inds, k=tf.shape(input=disable_bg_inds)[-1])
        disable_bg_inds = tf.reverse(disable_bg_inds, [0])
        disable_bg_inds = tf.compat.v1.sparse_to_dense(
            disable_bg_inds, tf.shape(input=labels, out_type=tf.int64),
            True, default_value=False
        )
        # Put -1 to ignore the anchors in the selected indices
        return tf.compat.v1.where(
            condition=tf.squeeze(disable_bg_inds),
            x=tf.cast(tf.fill(tf.shape(input=labels), -1), dtype=tf.float32), y=labels
        )

    # Recalculate the foreground indices after (maybe) disable some of them

    # Get foreground indices, get True in the indices where we have a one.
    fg_inds = tf.equal(labels, 1)
    # We get only the indices where we have True.
    fg_inds = tf.squeeze(tf.compat.v1.where(fg_inds), axis=1)
    fg_inds_size = tf.size(input=fg_inds)

    num_bg = tf.cast(minibatch_size - fg_inds_size, dtype=tf.int32)
    # Get background indices, get True in the indices where we have a zero.
    bg_inds = tf.equal(labels, 0)
    # We get only the indices where we have True.
    bg_inds = tf.squeeze(tf.compat.v1.where(bg_inds), axis=1)
    bg_inds_size = tf.size(input=bg_inds)
    # Condition for check if we have too many positive labels.
    subsample_negative_cond = bg_inds_size > num_bg
    # Check the condition and subsample positive labels.
    labels = tf.cond(
        pred=subsample_negative_cond,
        true_fn=subsample_negative, false_fn=lambda: labels
    )

    # Return bbox targets with shape (anchors.shape[0], 4).

    # Find the closest gt box for each anchor.
    argmax_overlaps = tf.argmax(input=overlaps, axis=1)
    # Eliminate duplicates.
    argmax_overlaps_unique, _ = tf.unique(argmax_overlaps)
    # Filter the gt_boxes.
    # We get only the indices where we have "inside anchors".
    anchor_filter_inds = tf.compat.v1.where(anchor_filter)
    gt_boxes = tf.gather(gt_boxes, argmax_overlaps)

    bbox_targets = encode_tf(anchors, gt_boxes)

    # For the anchors that arent foreground, we ignore the bbox_targets.
    anchor_foreground_filter = tf.equal(labels, 1)
    bbox_targets = tf.compat.v1.where(
        condition=anchor_foreground_filter,
        x=bbox_targets, y=tf.zeros_like(bbox_targets)
    )

    # We unroll "inside anchors" value for all anchors (for shape
    # compatibility).

    # We complete the missed indices with zeros
    # (because scatter_nd has zeros as default).
    bbox_targets = tf.scatter_nd(
        indices=tf.cast(anchor_filter_inds, dtype=tf.int32),
        updates=bbox_targets,
        shape=tf.shape(input=all_anchors)
    )

    labels_scatter = tf.scatter_nd(
        indices=tf.cast(anchor_filter_inds, dtype=tf.int32),
        updates=labels,
        shape=[tf.shape(input=all_anchors)[0]]
    )
    # We have to put -1 to ignore the indices with 0 generated by
    # scatter_nd, otherwise it will be considered as background.
    labels = tf.compat.v1.where(
        condition=anchor_filter, x=labels_scatter,
        y=tf.cast(tf.fill(tf.shape(input=labels_scatter), -1), dtype=tf.float32)
    )

    max_overlaps = tf.scatter_nd(
        indices=tf.cast(anchor_filter_inds, dtype=tf.int32),
        updates=max_overlaps,
        shape=[tf.shape(input=all_anchors)[0]]
    )

    return labels, bbox_targets, max_overlaps
