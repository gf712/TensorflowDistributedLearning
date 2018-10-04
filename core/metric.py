import tensorflow as tf
import numpy as np

IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def mIOU(y_true,
         y_pred,
         metrics_collections=None,
         updates_collections=None,
         name="iou"):

    with tf.variable_scope(name):

        def iou_internal(y_true_internal, y_pred_internal):
            tp = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(y_true_internal, 1), tf.equal(y_pred_internal, 1))))
            fp = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(y_true_internal, 0), tf.equal(y_pred_internal, 1))))
            fn = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(y_true_internal, 1), tf.equal(y_pred_internal, 0))))

            denominator = tp + fp + fn

            score = tf.expand_dims(tf.cond(tf.greater(denominator, 0),
                                           lambda: tf.to_float(tp / denominator),
                                           lambda: 1.0), 0)

            return tf.reduce_mean(
                score * tf.to_float(tf.greater(tf.tile(score, [len(IOU_THRESHOLDS)]), IOU_THRESHOLDS)))

        scores = tf.map_fn(lambda x: iou_internal(x[0], x[1]), elems=(y_true, y_pred), dtype=tf.float32)

        # return tf.reduce_mean(scores)

        # based on https://stackoverflow.com/questions/47753736/custom-metrics-with-tf-estimator
        iou, update_op = tf.metrics.mean(scores)

        if metrics_collections:
            tf.add_to_collections(metrics_collections, iou)

        if updates_collections:
            tf.add_to_collections(updates_collections, update_op)

        return iou, update_op


def mean_accuracy(y_true,
                  y_pred,
                  metrics_collections=None,
                  updates_collections=None,
                  name="acc"):

    with tf.variable_scope(name):

        scores = tf.reduce_mean(tf.to_float(tf.equal(y_true, y_pred)), axis=[1, 2, 3])

        # return tf.reduce_mean(scores)
        acc, update_op = tf.metrics.mean(scores)

        if metrics_collections:
            tf.add_to_collections(metrics_collections, acc)

        if updates_collections:
            tf.add_to_collections(updates_collections, update_op)

        return acc, update_op
