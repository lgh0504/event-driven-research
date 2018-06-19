import tensorflow as tf


def attention_layer(batch_vector_buckets, unit_size, name):
    """
    add attention mechanism to a list of vector to get a single vector
    :param batch_vector_buckets: tensor with shape [batch_size, sample_num, input_size]
    :param unit_size: size of dense layer
    :param name: the name of this attention layer
    :return: a attention based vector
    """
    with tf.variable_scope(name):
        # use a dense layer to turn input from [batch_size, sample_num, input_size]
        # to [batch_size, sample_num, unit_size]
        h = tf.layers.dense(batch_vector_buckets, units=unit_size,
                            activation=tf.nn.sigmoid)
        # generate the alpha vector with shape [batch_size, sample_num, 1]
        alpha = tf.nn.softmax(tf.reduce_sum(h, axis=2, keep_dims=True), dim=1)

        # generate attention output, tf.multiply will broadcast a scale to the whole vector
        # then reduce sum all vectors, return a output with shape [batch_size, unit_size]
        attention_output = tf.reduce_sum(tf.multiply(h, alpha), axis=1)

        return attention_output
