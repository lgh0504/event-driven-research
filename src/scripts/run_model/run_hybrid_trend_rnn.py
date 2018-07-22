from __future__ import print_function
import os
import pickle
from os import path
import tensorflow as tf
from models.input_fn import basic_input_fn
from models.model_fn import hybrid_trend_rnn


def run_model(model_path, local_prices, global_prices, local_events):

    # prepare data
    target_piece = pickle.load(open(local_prices, "rb"))
    other_prices = pickle.load(open(global_prices, "rb"))
    texts = pickle.load(open(local_events, "rb"))
    labels = []
    for i in range(0, len(target_piece)):
        if i != 0:
            labels.append(target_piece[i][0])
    target_piece = target_piece[0:-1]
    other_prices = other_prices[0:-1]
    texts = texts[0:-1]

    # set up parameters
    params = {'length': 30,
              'batch_size': 10,
              'state_size': 1024,
              'drop_rate': 0.05,
              'attention_size': 512,
              'learning_rate': 0.05}

    # set up run config
    run_config = tf.estimator.RunConfig(save_summary_steps=100)

    # set the model
    classifier = tf.estimator.Estimator(
        model_fn=hybrid_trend_rnn,
        params=params,
        model_dir=model_path,
        config=run_config)

    classifier.train(lambda: basic_input_fn(event_texts=texts, target_price=target_piece,
                                            other_prices=other_prices, final_price=labels,
                                            repeat=10, batch=10), max_steps=100000)


if __name__ == "__main__":

    # set root path
    training_data_root = os.environ['TRAINING_DATA_ROOT']
    model_root = os.environ['MODEL_ROOT']

    # set model path
    model_path = path.join(model_root, "goog_basic_one/")

    # set data path
    local_prices = path.join(training_data_root, "resources/training_data/goog.pl")
    other_path = path.join(training_data_root, "resources/training_data/other.pl")
    text_path = path.join(training_data_root, "resources/training_data/goog_text_vec.pl")

    # run the model
    run_model(model_path, local_prices, other_path, text_path)
