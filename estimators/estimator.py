import tensorflow as tf


def estimator_spec_for_softmax_classification(logits, labels, mode, config):
    """Returns EstimatorSpec instance for softmax classification."""
    # TODO: Move to Trainer class
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
            })

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
    }

    summary_hook = tf.train.SummarySaverHook(
        config.SAVE_EVERY_N_STEPS,
        output_dir='/tmp/tf',
        summary_op=tf.summary.merge_all())

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, training_hooks=[summary_hook])
