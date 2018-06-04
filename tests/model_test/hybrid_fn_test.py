import random
import sys
import numpy as np
import tensorflow as tf
from os import path

current_path = path.dirname(path.abspath(__file__))
parent_path = path.dirname(current_path)
sys.path.append(parent_path)
from models.model_fn import complex_model

keep_rate = 0.9
max_time = 15
state_size = 5
num_classes = 1
batch_size = 5
X_size = 1
Y_size = 1
E_size = 1
L_size = 1
train_steps = 100
model_path = path.join(parent_path, 'resources/model_checkpoint/event_test')


def generate_data(number_of_batch):
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
                tempY.append(tempY[i - 1] + tempX[i - 1] * tempE[i - 1])

            label.append(tempY[max_time - 1] + tempX[max_time - 1] * tempE[max_time - 1])
            X.append(tempX)
            Y.append(tempY)
            event.append(tempE)
    train_data['X'] = np.array(train_data['X']).reshape([-1, max_time, 1])
    train_data['Y'] = np.array(train_data['Y']).reshape([-1, max_time, 1])
    train_data['event'] = np.array(train_data['event']).reshape([-1, max_time, 1])
    label = np.array(label).reshape([-1, 1])
    return train_data, label


x_train, y_train = generate_data(100)
x_test, y_test = generate_data(10)


def train_input_fn(features, labels, batch):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.batch(batch)


classifier = tf.estimator.Estimator(
    model_fn=complex_model,
    params={
        'batch_size': batch_size,
        'state_size': state_size,
        'truncated_backprop_length': max_time,
        'keep_rate': keep_rate,
        'label_size': num_classes
    },
    model_dir=model_path,
    config=tf.estimator.RunConfig().replace(save_summary_steps=5))

classifier.train(
    input_fn=lambda: train_input_fn(x_train, y_train, batch_size), steps=100)

eval_result = classifier.evaluate(
    input_fn=lambda: train_input_fn(x_test, y_test, batch_size))

print('\nTest set accuracy: {MSE:0.3f}\n'.format(**eval_result))
