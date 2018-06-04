import tensorflow as tf
import numpy as np
from models.building_blocks import dynamic_lstm
from models.building_blocks import attention_based_lstm


# shape of feature X: [batch_size, truncated_backprop_length, exogenous_number]
# shape of feature Y: [batch_size, truncated_backprop_length, target_dimension]
# shape of feature event: [batch_size, truncated_backprop_length, embedding_dimension]
# shape of label : [batch_size, label_dimension]
def complex_model(features, labels, mode, params):
    """

    :param features: a dict
    :param labels:
    :param mode:
    :param params:
    :return:
    """
    # declare all training data_methods
    X = features['X']
    Y = features['Y']
    events = features['event']
    Xt = tf.unstack(X, axis=1)
    Xn = tf.unstack(X, axis=2)

    # set parameters
    batch_size = params['batch_size']
    state_size = params['state_size']
    truncated_backprop_length = params['truncated_backprop_length']
    label_size = params['label_size']
    keep_rate = params['keep_rate']

    # first lstm for attention of exogenous
    lstm0 = dynamic_lstm.DynamicLstm(state_size=state_size, batch_size=batch_size, keep_rate=keep_rate,
                                     variable_scope="LSTM0")
    hidden_states = lstm0.run(X)
    attention_hidden_states = tf.unstack(hidden_states, axis=1)

    # apply attention to exogenous
    W0 = tf.Variable(np.random.rand(state_size, truncated_backprop_length), dtype=tf.float32)
    U0 = tf.Variable(np.random.rand(truncated_backprop_length, truncated_backprop_length), dtype=tf.float32)
    V0 = tf.Variable(np.random.rand(truncated_backprop_length, 1), dtype=tf.float32)

    weights = []
    for h in attention_hidden_states:
        attention_matrix = 0
        # x's shape [batch_size, max_time]
        for x in Xn:  # exogenous_number
            # e's shape [b, 1]
            e = tf.matmul(tf.nn.tanh(tf.matmul(h, W0) + tf.matmul(x, U0)), V0)
            if attention_matrix == 0:
                attention_matrix = e
            else:
                attention_matrix = tf.concat([attention_matrix, e], axis=1)
        attention_matrix = tf.nn.softmax(attention_matrix)
        weights.append(attention_matrix)

    weighted_Xt = []
    for i in range(0, truncated_backprop_length):
        weighted_Xt.append(weights[i] * Xt[i])

    # reshape the weighted_x
    weighted_Xt = tf.stack(weighted_Xt)
    weighted_Xt = tf.unstack(weighted_Xt, axis=1)
    weighted_Xt = tf.stack(weighted_Xt)

    # get all hidden vector again
    hidden_states, last_state = lstm0.run(weighted_Xt)

    # build second attention network
    lstm1 = attention_based_lstm.AttentionBasedLstm(state_size, batch_size, truncated_backprop_length, keep_rate)
    exogenous_hidden_states = lstm1.run(Y, hidden_states)

    # generate output from da-rnn
    exogenous_output = exogenous_hidden_states[-1]  # [batch_size, state_size]

    # generate event hidden_state
    lstm2 = dynamic_lstm.DynamicLstm(state_size, batch_size, keep_rate, variable_scope="LSTM1")
    event_hidden_states, _ = lstm2.run(events)

    # combine event_hidden_state and exogenous_hidden_states
    event_hidden_states = tf.unstack(event_hidden_states, axis=1)
    event_hidden_states = event_hidden_states[1:]
    Wr = tf.Variable(np.random.rand(state_size, state_size), dtype=tf.float32)
    Br = tf.Variable(np.random.rand(batch_size, 1), dtype=tf.float32)
    relation_matrix = 0
    for i in range(0, truncated_backprop_length):
        e = event_hidden_states[i] * (tf.matmul(exogenous_hidden_states[i], Wr) + Br)
        e = tf.reduce_sum(e, axis=1)
        e = tf.reshape(e, [batch_size, 1])
        if relation_matrix == 0:
            relation_matrix = e
        else:
            relation_matrix = tf.concat([relation_matrix, e], axis=1)
    relation_matrix = tf.nn.softmax(relation_matrix)

    relation_matrix = tf.unstack(relation_matrix, axis=1)
    event_output = 0  # c's shape [batch_size, m]
    for j in range(0, truncated_backprop_length):
        Beta = tf.reshape(relation_matrix[j], [batch_size, 1])
        if event_output == 0:
            event_output = Beta * event_hidden_states[j]
        else:
            event_output += Beta * event_hidden_states[j]

    # combine exogenous_output and event_output
    # event_output [batch_size, state_size]
    # exogenous_output [batch_size, state_size]
    combined_output = tf.concat([event_output, exogenous_output], axis=1)
    values = tf.layers.dense(combined_output, label_size)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'values': values}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.mean_squared_error(labels, values)

    # Compute evaluation metrics.
    accuracy = tf.metrics.mean_squared_error(labels=labels, predictions=values, name='MSE')
    metrics = {'MSE': accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
