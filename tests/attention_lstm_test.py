from os import path
import sys
import numpy as np
import tensorflow as tf
current_path = path.dirname(path.abspath(__file__))
parent_path = path.dirname(current_path)
sys.path.append(parent_path)
from models.building_blocks import attention_based_lstm

truncated_backprop_length = 3
state_size = 3
batch_size = 2

# shape [batch_size, max_time, input_dimension]
input_series = [[[0.,0.,0.], [1.,1.,1.], [2.,2.,2.]], [[3.,3.,3.], [4.,4.,4.], [5.,5.,5.]]]
# shape [batch_size, max_time, state_size]
attention_weights = [[[0.,0.,0.], [1.,1.,1.], [2.,2.,2.]], [[3.,3.,3.], [4.,4.,4.], [5.,5.,5.]]]

attention_lstm = attention_based_lstm.AttentionBasedLstm(state_size, batch_size, truncated_backprop_length, 0.9)

hidden_states = attention_lstm.run(input_series, attention_weights)

# test shape
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(hidden_states))