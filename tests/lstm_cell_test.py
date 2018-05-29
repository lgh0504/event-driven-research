import tensorflow as tf

batch_size = 2
state_size = 3


init_cell_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
init_hidden_state = tf.constant(1, dtype=tf.float32, shape=[batch_size, state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(init_cell_state, init_hidden_state)
cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)

# [batch_size, ?]
x = tf.constant([[1.,2.,3.,4.], [5.,6.,7.,8.]])
output, state = cell(x, init_state)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run([output, state]))
