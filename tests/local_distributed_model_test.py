import random
import sys
import numpy as np
import tensorflow as tf
from os import path
import os
import json

current_path = path.dirname(path.abspath(__file__))
parent_path = path.dirname(current_path)
sys.path.append(parent_path)
from models import lstm_fn

num_epochs = 10
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
input_dimension = 1
train_steps = 100
model_path = path.join(parent_path, 'resources/model_checkpoint/distributed_test')


def generate_data(number_of_batch):
    data = []
    label = []
    for _ in range(0, number_of_batch):
        for __ in range(0, batch_size):
            p = random.random()
            if p > 0.5:
                q = random.random()
                for index in range(0, truncated_backprop_length):
                    data.append(np.float32(q + 0.1 * index))
                label.append(1)
            else:
                q = random.random()
                for index in range(0, truncated_backprop_length):
                    data.append(np.float32(q - 0.1 * index))
                label.append(0)

    data = np.array(data)
    label = np.array(label)
    data = data.reshape(-1, truncated_backprop_length, input_dimension)
    return data, label


x_train, y_train = generate_data(100)
x_test, y_test = generate_data(10)
x_train = {'feature': x_train}
x_test = {'feature': x_test}


def train_input_fn(features, labels, batch):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.batch(batch)


config = {"cluster": {'ps':     ['localhost:2222'],
                      'chief': ['localhost:2223']}}
if sys.argv[1] == "chief":
    print("I am chief")
    config["task"] = {'type': 'chief', 'index': 0}
else:
    print("I am ps")
    config["task"] = {'type': 'ps', 'index': 0}

os.environ['TF_CONFIG'] = json.dumps(config)

classifier = tf.estimator.Estimator(
    model_fn=lstm_fn.lstm_model_fn,
    params={
        'batch_size': batch_size,
        'state_size': state_size,
        'truncated_backprop_length': truncated_backprop_length,
        'input_dimension': input_dimension,
        'num_classes': num_classes
    },
    model_dir=model_path,
    config=tf.estimator.RunConfig().replace(save_summary_steps=5))

train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(x_train, y_train, batch_size), max_steps=100)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(x_test, y_test, batch_size))

tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
