from os import path
import sys
import numpy as np
import tensorflow as tf
current_path = path.dirname(path.abspath(__file__))
parent_path = path.dirname(current_path)
sys.path.append(parent_path)
from models.building_blocks import dynamic_lstm


num_epochs = 100
total_series_length = 5000
truncated_backprop_length = 2
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length

# generate test data_methods
x = np.random.rand(batch_size * truncated_backprop_length)
x = np.array(x.tolist(), dtype='f')
x = x.reshape((batch_size, truncated_backprop_length, 1))

# build network
lstm = dynamic_lstm.dynamic_lstm(state_size, batch_size, 0.9)
hidden_states, last_state = lstm.run(x)
hidden_states = tf.unstack(hidden_states, axis=1)
attention_hidden_states = hidden_states[0:-1]


# test shape
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(attention_hidden_states))
