import tensorflow as tf
from models.building_blocks import DynamicLstm
from models.building_blocks import AttentionLayer


def basic_model_one(features, labels, mode, params):
    """
    my basic hybrid model one with simple approach
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
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
    target_price = features['target_price']  # [b, n, 1]
    outer_prices = features['other_prices']  # [b, n, m]
    event_texts = features['event_texts']  # [b, n, d]

    ''' building basic model structure'''
    # inner trend capture
    inner_trend_capture = DynamicLstm(batch_size=batch_size, state_size=state_size,
                                      keep_rate=1 - drop_rate, variable_scope="inner_price_capture")
    inner_trend_h_vec = inner_trend_capture.run(target_price)  # [b, n, h]

    # event trend capture
    event_trend_capture = DynamicLstm(batch_size=batch_size, state_size=state_size,
                                      keep_rate=1 - drop_rate, variable_scope="event_trend_capture")
    event_trend_h_vec = event_trend_capture.run(event_texts)  # [b, n, h]

    # outer trend capture
    outer_trend_capture = DynamicLstm(batch_size=batch_size, state_size=state_size,
                                      keep_rate=1 - drop_rate, variable_scope="outer_trend_capture")
    outer_trend_h_vec = outer_trend_capture.run(outer_prices)  # [b, n, h]

    ''' building cross trend attention '''
    # building attention weights to inner trend
    inner_weights = 0
    attention_dense = tf.layers.Dense(units=1, name="inner_weight_dense")
    inner_attention = AttentionLayer(attention_size=attention_size, name='inner_attention')
    for i in range(0, length):
        attention_h = tf.slice(inner_trend_h_vec, [0, i, 0], [batch_size, 1, state_size])  # [b, 1, h ]
        attention_h = tf.tile(attention_h, [1, length, 1])  # [b, n, h ]
        attention_h = tf.concat([outer_trend_h_vec, attention_h], axis=2)  # [b, n, 2h]
        attention_h = inner_attention.run(attention_h)  # [b, u]
        attention_h = attention_dense(attention_h)  # [b, 1]
        attention_h = tf.reshape(attention_h, [batch_size, 1, 1])  # [b, 1, 1]
        if i == 0:
            inner_weights = attention_h
        else:
            inner_weights = tf.concat([inner_weights, attention_h], axis=1)  # [b, n, 1]
    inner_trend_h_vec = tf.multiply(inner_weights, inner_trend_h_vec)

    # building attention weights to event trend
    event_weights = 0
    attention_dense = tf.layers.Dense(units=1, name="event_weight_dense")
    event_attention = AttentionLayer(attention_size=attention_size, name='event_attention')
    for i in range(0, length):
        attention_h = tf.slice(event_trend_h_vec, [0, i, 0], [batch_size, 1, state_size])  # [b, 1, h ]
        attention_h = tf.tile(attention_h, [1, length, 1])  # [b, n, h ]
        attention_h = tf.concat([outer_trend_h_vec, attention_h], axis=2)  # [b, n, 2h]
        attention_h = event_attention.run(attention_h)  # [b, u]
        attention_h = attention_dense(attention_h)  # [b, 1]
        attention_h = tf.reshape(attention_h, [batch_size, 1, 1])  # [b, 1, 1]
        if i == 0:
            event_weights = attention_h
        else:
            event_weights = tf.concat([event_weights, attention_h], axis=1)  # [b, n, 1]
    event_trend_h_vec = tf.multiply(event_weights, event_trend_h_vec)

    ''' combine inner trend and event trend'''
    hybrid_trend = tf.concat([inner_trend_h_vec, event_trend_h_vec], axis=2)  # [b, n, 2h]
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

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
