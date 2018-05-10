import tensorflow as tf
import numpy as np


# a homemade lstm network
class CustomizedLstm:

    # input data_methods has the shape of [batch_size, time_size, feature_size]
    def __init__(self, state_size, batch_size, time_size, feature_size):
        # set up parameters
        self._state_size = state_size
        self._batch_size = batch_size
        self._time_size = time_size
        self._feature_size = feature_size

        ''' set up network structure'''
        # set up init state
        self._init_cell_state = tf.constant(0, dtype=tf.float32, shape=[state_size, self._batch_size])
        self._init_hidden_state = tf.constant(0, dtype=tf.float32, shape=[state_size, self._batch_size])

        # set up network parameters
        self._Wf, self._Bf = self._get_w_b()
        self._Wi, self._Bi = self._get_w_b()
        self._Wo, self._Bo = self._get_w_b()
        self._Ws, self._Bs = self._get_w_b()

    # helper method to set up network parameter
    def _get_w_b(self):
        return tf.Variable(np.random.rand(self._state_size, self._state_size + self._feature_size),
                           dtype=tf.float32), \
               tf.Variable(np.zeros((self._state_size, 1)), dtype=tf.float32)

    # run for an input series
    # input_series has a shape of [batch_size, time_size, feature_size]
    # return all cell states & hidden_states [batch_size, time_size, feature_size]
    def run(self, input_series):
        current_cell_state = self._init_cell_state
        current_hidden_state = self._init_hidden_state
        cell_states = []
        hidden_states = []

        # unstack the input series to shape [time_size, batch_size, feature_size]
        input_series = tf.unstack(input_series, axis=1)

        # run a for loop
        for current_input in input_series:
            current_input = tf.transpose(current_input)

            # Increasing number of row
            input_and_state_concatenated = tf.concat([current_input, current_hidden_state], axis=0)

            # update parameters
            forget_gate = tf.sigmoid(tf.matmul(self._Wf, input_and_state_concatenated) + self._Bf)
            input_gate = tf.sigmoid(tf.matmul(self._Wi, input_and_state_concatenated) + self._Bi)
            output_gate = tf.sigmoid(tf.matmul(self._Wo, input_and_state_concatenated) + self._Bo)

            next_cell_state = forget_gate * current_cell_state + input_gate * \
                              tf.tanh(tf.matmul(self._Ws, input_and_state_concatenated) + self._Bs)
            next_hidden_state = output_gate * tf.tanh(next_cell_state)

            # update states
            cell_states.append(next_cell_state)
            hidden_states.append(next_hidden_state)
            current_cell_state = next_cell_state
            current_hidden_state = next_hidden_state

        return cell_states, hidden_states
