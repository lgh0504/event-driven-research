import tensorflow as tf
from models.building_blocks import DynamicLstm
from models.building_blocks import AttentionLayer

def local_trend_rnn(features, labels, mode, params):
    """
    local trend model only using information about the target stock (price & events)
    :param features: [local_prices] and [local_events]
    :param labels: price result
    :param mode: is the running mode
    :param params: are model param settings
    :return: saving model in the config path
    """

    ''' set up variables & parameters'''
    # get parameters & super parameters
    length = params['length']
    batch_size = params['batch_size']
    state_size = params['state_size']
    drop_rate = params['drop_rate']
    attention_size = params['attention_size']
    learning_rate = params['learning_rate']

    # get training features and label
    local_prices = features['local_prices']  # [b, n, 1]
    local_events = features['local_events']    # [b, n, d]

    ''' building basic model structure'''
    # inner trend capture
    price_trend_capture = DynamicLstm(batch_size=batch_size, state_size=state_size,
                                      keep_rate=1 - drop_rate, variable_scope="price_trend_capture")
    price_trend_h_vec = price_trend_capture.run(local_prices)  # [b, n, h]

    # event trend capture
    event_trend_capture = DynamicLstm(batch_size=batch_size, state_size=state_size,
                                      keep_rate=1 - drop_rate, variable_scope="event_trend_capture")
    event_trend_h_vec = event_trend_capture.run(local_events)  # [b, n, h]

    ''' combine price trend and event trend '''
    hybrid_trend = tf.concat([price_trend_h_vec, event_trend_h_vec], axis=2)  # [b, n, 2h]
    hybrid_trend_capture = DynamicLstm(batch_size=batch_size, state_size=state_size,
                                       keep_rate=1 - drop_rate, variable_scope="hybrid_trend_capture")
    hybrid_trend_h_vec = hybrid_trend_capture.run(hybrid_trend)  # [b, n, h]

    ''' generate final prediction '''
    output_dense = tf.layers.Dense(units=1, name="output_dense")
    hybrid_attention = AttentionLayer(attention_size=attention_size, name="hybrid_attention")
    values = output_dense(hybrid_attention.run(hybrid_trend_h_vec))  # [b, 1]

    ''' config '''
    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'values': values}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.mean_squared_error(labels, values)

    # Compute evaluation metrics.
    mse = tf.metrics.mean_squared_error(labels=labels, predictions=values, name='MSE')
    metrics = {'MSE': mse}
    tf.summary.scalar('mean_squared_error', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

