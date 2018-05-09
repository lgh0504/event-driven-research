import tensorflow as tf
import numpy as np
batch_size = 2
state_size = 3
truncated_backprop_length = 4
X = [
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0,  2.0], [3.0, 3.0, 3.0]],
    [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]]
]

attention_hidden_states = [
    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
]

Xt = tf.unstack(X, axis=1)
Xn = tf.unstack(X, axis=2)


# first attention network
W0 = tf.Variable(np.random.rand(state_size, truncated_backprop_length), dtype=tf.float32)
U0 = tf.Variable(np.random.rand(truncated_backprop_length, truncated_backprop_length), dtype=tf.float32)
V0 = tf.Variable(np.random.rand(truncated_backprop_length, 1), dtype=tf.float32)

# build exogenous attention
weights = []
for h in attention_hidden_states:
    attention_matrix = 0
    # x's shape [batch_size, max_time]
    for x in Xn:  # exogenous_number
        # e's shape [b, 1]
        e = tf.matmul(tf.nn.tanh(tf.matmul(h, W0) + tf.matmul(x, U0)), V0)
        if attention_matrix == 0:
            attention_matrix = e
        else:
            attention_matrix = tf.concat([attention_matrix, e], axis=1)
    attention_matrix = tf.nn.softmax(attention_matrix)
    weights.append(attention_matrix)

weighted_Xt = []
for i in range(0,truncated_backprop_length):
    weighted_Xt.append(weights[i] * Xt[i])

weighted_Xt = tf.stack(weighted_Xt)
weighted_Xt = tf.unstack(weighted_Xt, axis=1)
weighted_Xt = tf.stack(weighted_Xt)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(weighted_Xt))