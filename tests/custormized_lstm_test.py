from __future__ import print_function, division

import sys
import numpy as np
import tensorflow as tf
from os import path

current_path = path.dirname(path.abspath(__file__))
parent_path = path.dirname(current_path)
sys.path.append(parent_path)
from models.building_blocks import custormized_lstm

num_epochs = 100
total_series_length = 5000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length

# generate data_methods
x = np.random.rand(batch_size * truncated_backprop_length)
x = np.array(x.tolist(), dtype='f')
x = x.reshape((batch_size, truncated_backprop_length, 1))

# construct data_methods map
lstm = custormized_lstm.CustomizedLstm(state_size, batch_size, truncated_backprop_length, 1)
cell_states, hidden_states = lstm.run(x)

final_hidden_state = hidden_states[-1]

W = tf.Variable(np.random.rand(num_classes, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((num_classes, 1)), dtype=tf.float32)

# Compute logits
logits = tf.matmul(W, final_hidden_state) + b
logits = tf.transpose(logits)
predicted_classes = tf.argmax(logits, 1)
class_ids = predicted_classes[:, tf.newaxis],
probabilities = tf.nn.softmax(logits)

# start session
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(probabilities))
