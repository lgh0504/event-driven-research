import tensorflow as tf
import numpy as np


# an attention based lstm built on dynamic rnn
class AttentionBasedLstm:
    def __init__(self, state_size, batch_size, truncated_backprop_length, keep_rate):
        # build the lstm network
        self._batch_size = batch_size
        self._state_size = state_size
        self._truncated_backprop_length = truncated_backprop_length
        self._init_cell_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
        self._init_hidden_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
        self._init_state = tf.nn.rnn_cell.LSTMStateTuple(self._init_cell_state, self._init_hidden_state)
        self._cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        self._cell = tf.nn.rnn_cell.DropoutWrapper(self._cell, output_keep_prob=keep_rate)

        # attention network
        self._W = tf.Variable(np.random.rand(state_size, state_size), dtype=tf.float32)
        self._U = tf.Variable(np.random.rand(state_size, state_size), dtype=tf.float32)
        self._V = tf.Variable(np.random.rand(state_size, 1), dtype=tf.float32)

    # run the dynamic lstm with attention
    # input series: [batch_size, truncated_backprop_length, input_dimension]
    # attention_weights: [batch_size, truncated_backprop_length, state_size]
    # output two variables: hidden_states, last_state
    # hidden_state shape [batch_size, max_time, state_size]
    # last_state shape [batch_size, state_size]
    def run(self, input_series, attention_weights):
        # shape [truncated_backprop_length, batch_size, input_dimension]
        input_series = tf.unstack(input_series, axis=1)
        # shape [truncated_backprop_length, batch_size, input_dimension]
        attention_weights = tf.unstack(attention_weights, axis=1)

        attention_weights = attention_weights[1:]
        current_state = self._init_state
        hidden_state = self._init_hidden_state

        hidden_states = []
        for i in range(0, self._truncated_backprop_length):

            # build attention softmax
            attention_matrix = 0
            for h in attention_weights:
                e = tf.matmul(tf.nn.tanh(tf.matmul(hidden_state, self._W) + tf.matmul(h, self._U)), self._V)
                if attention_matrix == 0:
                    attention_matrix = e
                else:
                    attention_matrix = tf.concat([attention_matrix, e], axis=1)
            attention_matrix = tf.nn.softmax(attention_matrix)

            # build input c
            attention_matrix = tf.unstack(attention_matrix, axis=1)
            c = 0  # c's shape [batch_size, m]
            for j in range(0, self._truncated_backprop_length):
                Beta = tf.reshape(attention_matrix[j], [self._batch_size, 1])
                if c == 0:
                    c = Beta * attention_weights[j]
                else:
                    c += Beta * attention_weights[j]

            # concat c & input
            c = tf.concat([c, input_series[i]], axis=1)
            output, current_state = self._cell(c, current_state)
            hidden_state = output
            hidden_states.append(output)

        return hidden_states




