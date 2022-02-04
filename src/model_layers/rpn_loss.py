from utils.losses import smooth_l1_loss
import tensorflow as tf
def RPNLoss(prediction_dict):
    """
    Returns cost for Region Proposal Network based on:
    Args:
        rpn_cls_score: Score for being an object or not for each anchor
            in the image. Shape: (num_anchors, 2)
        rpn_cls_target: Ground truth labeling for each anchor. Should be
            * 1: for positive labels
            * 0: for negative labels
            * -1: for labels we should ignore.
            Shape: (num_anchors, )
        rpn_bbox_target: Bounding box output delta target for rpn.
            Shape: (num_anchors, 4)
        rpn_bbox_pred: Bounding box output delta prediction for rpn.
            Shape: (num_anchors, 4)
    Returns:
        Multiloss between cls probability and bbox target.
    """

    rpn_cls_score = prediction_dict['rpn_cls_score']
    rpn_cls_target = prediction_dict['rpn_cls_target']

    rpn_bbox_target = prediction_dict['rpn_bbox_target']
    rpn_bbox_pred = prediction_dict['rpn_bbox_pred']

    with tf.compat.v1.variable_scope('RPNLoss'):
        # Flatten already flat Tensor for usage as boolean mask filter.
        rpn_cls_target = tf.cast(tf.reshape(
            rpn_cls_target, [-1]), tf.int32, name='rpn_cls_target')
        # Transform to boolean tensor mask for not ignored.
        labels_not_ignored = tf.not_equal(
            rpn_cls_target, -1, name='labels_not_ignored')

        # Now we only have the labels we are going to compare with the
        # cls probability.
        labels = tf.boolean_mask(tensor=rpn_cls_target, mask=labels_not_ignored)
        cls_score = tf.boolean_mask(tensor=rpn_cls_score, mask=labels_not_ignored)

        # We need to transform `labels` to `cls_score` shape.
        # convert [1, 0] to [[0, 1], [1, 0]] for ce with logits.
        cls_target = tf.one_hot(labels, depth=2)

        # Equivalent to log loss
        ce_per_anchor = tf.nn.softmax_cross_entropy_with_logits(
            labels=cls_target, logits=cls_score
        )
        prediction_dict['cross_entropy_per_anchor'] = ce_per_anchor

        # Finally, we need to calculate the regression loss over
        # `rpn_bbox_target` and `rpn_bbox_pred`.
        # We use SmoothL1Loss.
        rpn_bbox_target = tf.reshape(rpn_bbox_target, [-1, 4])
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

        # We only care for positive labels (we ignore backgrounds since
        # we don't have any bounding box information for it).
        positive_labels = tf.equal(rpn_cls_target, 1)
        rpn_bbox_target = tf.boolean_mask(tensor=rpn_bbox_target, mask=positive_labels)
        rpn_bbox_pred = tf.boolean_mask(tensor=rpn_bbox_pred, mask=positive_labels)

        # We apply smooth l1 loss as described by the Fast R-CNN paper.
        reg_loss_per_anchor = smooth_l1_loss(
            rpn_bbox_pred, rpn_bbox_target, sigma=3.0
        )

        prediction_dict['reg_loss_per_anchor'] = reg_loss_per_anchor

        # Loss summaries.
        tf.compat.v1.summary.scalar('batch_size', tf.shape(input=labels)[0], ['rpn'])
        foreground_cls_loss = tf.boolean_mask(
            tensor=ce_per_anchor, mask=tf.equal(labels, 1))
        background_cls_loss = tf.boolean_mask(
            tensor=ce_per_anchor, mask=tf.equal(labels, 0))
        tf.compat.v1.summary.scalar(
            'foreground_cls_loss',
            tf.reduce_mean(input_tensor=foreground_cls_loss), ['rpn'])
        tf.compat.v1.summary.histogram(
            'foreground_cls_loss', foreground_cls_loss, ['rpn'])
        tf.compat.v1.summary.scalar(
            'background_cls_loss',
            tf.reduce_mean(input_tensor=background_cls_loss), ['rpn'])
        tf.compat.v1.summary.histogram(
            'background_cls_loss', background_cls_loss, ['rpn'])
        tf.compat.v1.summary.scalar(
            'foreground_samples', tf.shape(input=rpn_bbox_target)[0], ['rpn'])

        return {
            'rpn_cls_loss': tf.reduce_mean(input_tensor=ce_per_anchor),
            'rpn_reg_loss': tf.reduce_mean(input_tensor=reg_loss_per_anchor),
}