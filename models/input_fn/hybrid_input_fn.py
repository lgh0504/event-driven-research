# import pickle
# import numpy as np
# def hybrid_train_input():
#     # total_length = 1955
#     # event_dimension = 900
#     # max_time = 5
#     # target_stock = "aal"
#     # stock_pickle_path = ""
#     # text_pickle_path = ""
#     # stock_infos = pickle.load(open(stock_pickle_path, "rb"))   # a dict of data frame
#     # event_vectors = pickle.load(open(text_pickle_path), "rb")  # shape [total_length, event_dimension]
#     # target_price = stock_infos[target_stock]["open"].tolist()
#     # other_price =
#     features = {}
#     features['X'] =
#     features['Y'] = np.array(target_price).reshape([-1, max_time, 1])
#     features['event'] = np.array(event_vectors).reshape([-1, max_time, event_dimension])
#     labels =
#
#
#     return 0

import random

import numpy as np
import tensorflow as tf
import math


def relationship(y, x, e):
    return np.float32(math.cos(y) + math.sin(x * e))


def _generate_data(number_of_batch, max_time, batch_size):
    train_data = {}
    X = []
    Y = []
    event = []
    label = []
    train_data['X'] = X
    train_data['Y'] = Y
    train_data['event'] = event
    for _ in range(0, number_of_batch):
        for __ in range(0, batch_size):
            x0 = np.float32(random.random())
            y0 = np.float32(random.random())
            e0 = np.float32(random.random())
            tempX = []
            tempY = []
            tempE = []
            tempX.append(x0)
            tempY.append(y0)
            tempE.append(e0)
            for i in range(1, max_time):
                tempX.append(np.float32(random.random()))
                tempE.append(np.float32(random.random()))
                tempY.append(relationship(tempY[i - 1], tempX[i - 1], tempE[i - 1]))

            label.append(relationship(tempY[max_time - 1], tempX[max_time - 1], tempE[max_time - 1]))
            X.append(tempX)
            Y.append(tempY)
            event.append(tempE)
    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(Y))
    assert not np.any(np.isnan(event))
    assert not np.any(np.isnan(label))
    train_data['X'] = np.array(train_data['X']).reshape([-1, max_time, 1])
    train_data['Y'] = np.array(train_data['Y']).reshape([-1, max_time, 1])
    train_data['event'] = np.array(train_data['event']).reshape([-1, max_time, 1])
    label = np.array(label).reshape([-1, 1])
    return train_data, label


def train_input_fn(number_of_batch, max_time, batch_size):
    """An input function for training"""
    features, labels = _generate_data(number_of_batch, max_time, batch_size)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.repeat(1).batch(batch_size)


def eval_input_fn(number_of_batch, max_time, batch_size):
    """An input function for training"""
    features, labels = _generate_data(number_of_batch, max_time, batch_size)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.batch(batch_size)
