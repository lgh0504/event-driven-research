import tensorflow as tf


class DynamicLstm:
    """
    a dynamic lstm built on tensorflow official API
    """

    def __init__(self, batch_size, state_size, keep_rate, variable_scope):
        """
        initialize a lstm
        :param batch_size: batch size
        :param state_size: size of hidden vector
        :param keep_rate: 1 - drop out rate
        :param variable_scope: the name scope of this LSTM
        """
        # set up parameters
        self._batch_size = batch_size
        self._state_size = state_size
        self._name_scope = variable_scope

        '''set up the network'''
        # set up init states
        # TODO: init state can be improved
        self._init_cell_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
        self._init_hidden_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
        self._init_state = tf.nn.rnn_cell.LSTMStateTuple(self._init_cell_state, self._init_hidden_state)

        # set up the lstm cell with dropout
        self._cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        self._cell = tf.nn.rnn_cell.DropoutWrapper(self._cell, output_keep_prob=keep_rate)

    #
    # output two variables: hidden_states, last_state
    # hidden_state shape [batch_size, max_time + 1, state_size]
    # last_state shape [batch_size, state_size]
    def run(self, input_series):
        """
        run the dynamic lstm
        :param input_series: input shape [batch_size, max_time, input_size]
        :return:

        """
        hidden_states, last_state = tf.nn.dynamic_rnn(self._cell, input_series,
                                                      initial_state=self._init_state,
                                                      dtype=tf.float32,
                                                      scope=self._name_scope)
        _init_state_concat = tf.reshape(self._init_hidden_state, [self._batch_size, 1, self._state_size])
        return tf.concat([_init_state_concat, hidden_states], axis=1), last_state
