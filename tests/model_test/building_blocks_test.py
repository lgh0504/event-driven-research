import tensorflow as tf
from models import building_blocks as bb


def attention_layer_test():
    batch = tf.constant(1., shape=[2, 3, 4], dtype=tf.float32)
    attention_output = bb.attention_layer(batch, 4, "test")

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(tf.shape(attention_output)))


if __name__ == "__main__":
    attention_layer_test()

