import tensorflow as tf
from models.building_blocks import attention_layer


def basic_predictor_fn(features, labels, mode, params):
    """
    A basic predictor function, using time irrelevant data and applying
    attention mechanism in news level
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """

    """ model structure """
    # apply attention to news first
    labels = tf.argmax(labels, 1)
    news_bucket = features['news_bucket']
    attention_news = attention_layer(news_bucket, params['attention_unit_size'], "news_level_attention")

    # connected to several dense layer
    net = attention_news
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # output layer
    logits = tf.layers.dense(net, units=params['n_classes'], activation=None)

    """ training setting """
    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
