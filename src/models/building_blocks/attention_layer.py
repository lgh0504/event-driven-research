import tensorflow as tf


class AttentionLayer:
    def __init__(self, attention_size, name):
        """
        use attention mechanism to merge vectors
        :param attention_size: size of dense layer
        :param name: the name of this attention layer
        :return: a new vector
        """
        self.dense = tf.layers.Dense(units=attention_size, activation=tf.nn.tanh, name=name)

    def run(self, vectors):
        """
        merge all vectors to one using attention mechanism
        :param vectors: [vector]
        :return: single vector
        """
        # from [b,n,d] to [b,n,d']
        h = self.dense(vectors)

        # to [b, n, 1]
        alpha = tf.nn.softmax(tf.reduce_sum(h, axis=2, keep_dims=True), dim=1)

        # generate attention output, tf.multiply will broadcast a scale to the whole vector
        # then reduce sum all vectors, return a output with shape [b, u]
        attention_output = tf.reduce_sum(tf.multiply(h, alpha), axis=1)

        return attention_output
