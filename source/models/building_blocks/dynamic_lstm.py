import tensorflow as tf


# a dynamic lstm built on tensorflow official API
class dynamic_lstm:
    def __init__(self, state_size, batch_size, keep_rate):
        # set up parameters
        self._batch_size = batch_size
        self._state_size = state_size

        ''' set up the network structure'''
        # set up init states
        self._init_cell_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
        self._init_hidden_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
        self._init_state = tf.nn.rnn_cell.LSTMStateTuple(self._init_cell_state, self._init_hidden_state)

        # set up the lstm cell with dropout
        self._cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        self._cell = tf.nn.rnn_cell.DropoutWrapper(self._cell, output_keep_prob=keep_rate)

    # run the dynamic lstm
    # input shape [batch_size, max_time, input_dimension]
    # output two variables: hidden_states, last_state
    # hidden_state shape [batch_size, max_time + 1, state_size]
    # last_state shape [batch_size, state_size]
    def run(self, input_series):
        outputs, last_state = tf.nn.dynamic_rnn(self._cell, input_series,
                                                initial_state=self._init_state,
                                                dtype=tf.float32)
        _init_state_concat = tf.reshape(self._init_hidden_state, [self._batch_size, 1, self._state_size])
        return tf.concat([_init_state_concat, outputs], axis=1), last_state
