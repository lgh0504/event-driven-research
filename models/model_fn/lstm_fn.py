import tensorflow as tf
import numpy as np
from os import path
import sys

current_path = path.dirname(path.abspath(__file__))
parent_path = path.dirname(current_path)
sys.path.append(parent_path)
from models.building_blocks import custormized_lstm


# lstm model fn
def lstm_model_fn(features, labels, mode, params):
    """LSTM model with a single lstm layer"""

    # create input tensor
    # input_series's shape [batch_size, time_step, input_dimension]
    # label_series's shape [batch_size, 1]
    input_series = features['feature']

    # build compute network
    batch_size = params['batch_size']
    state_size = params['state_size']
    truncated_backprop_length = params['truncated_backprop_length']
    input_dimension = params['input_dimension']
    num_classes = params['num_classes']

    lstm = custormized_lstm.CustomizedLstm(state_size=state_size, batch_size=batch_size,
                                           time_size=truncated_backprop_length,
                                           feature_size=input_dimension)

    cell_states, hidden_states = lstm.run(input_series)

    # final_hidden_states's shape [state_size, batch_size]
    final_hidden_state = hidden_states[-1]

    W = tf.Variable(np.random.rand(num_classes, state_size), dtype=tf.float32)
    b = tf.Variable(np.zeros((num_classes, 1)), dtype=tf.float32)

    # Compute logits
    logits = tf.matmul(W, final_hidden_state) + b
    logits = tf.transpose(logits)

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

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
