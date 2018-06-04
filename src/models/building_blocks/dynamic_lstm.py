import tensorflow as tf


class DynamicLstm:
    """ a dynamic lstm built on tensorflow official API """

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

        # set up init states
        # TODO: init state can be improved
        self._init_cell_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
        self._init_hidden_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
        self._init_state = tf.nn.rnn_cell.LSTMStateTuple(self._init_cell_state, self._init_hidden_state)

        # set up the lstm cell with dropout
        self._cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        self._cell = tf.nn.rnn_cell.DropoutWrapper(self._cell, output_keep_prob=keep_rate)

    def run(self, input_series):
        """
        run the dynamic lstm
        :param input_series: input shape [batch_size, max_time, input_size]
        :return: all hidden states shape [batch_size, max_time, input_size]
        """
        hidden_states, last_state = tf.nn.dynamic_rnn(self._cell, input_series,
                                                      initial_state=self._init_state,
                                                      dtype=tf.float32,
                                                      scope=self._name_scope)
        return hidden_states
