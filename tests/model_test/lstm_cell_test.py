import tensorflow as tf
from models.building_blocks import AttentionLayer
v1 = [[[1., 1., 1.], [2., 2., 2.], [0., 0., 0.]],
      [[3., 3., 3.], [4., 4., 4.], [0., 0., 0.]]]
v2 = tf.slice(v1, [0, 1, 0], [2, 1, 3])
v3 = tf.tile(v2, [1, 3, 1])
v4 = tf.concat([v1, v3], axis=2)
a = AttentionLayer(3, "aaa")
v5 = a.run(v4)
weight_dense = tf.layers.Dense(units=1, name="inner_weight_dense")
v6 = weight_dense(v5)
v7 = tf.reshape(v6, [2, 1, 1])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run([v7]))
