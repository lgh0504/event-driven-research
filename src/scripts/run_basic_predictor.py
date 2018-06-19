import os
from os import path
import tensorflow as tf
from models.model_fn import basic_predictor_fn
from models.input_fn import time_irrelevant_train_fn

if __name__ == "__main__":
    root_path = os.environ['EDR_ROOT_PATH']

    # set up path
    data_path = path.join(root_path, "resources/training_data/training_dict.pickle")
    model_path = path.join(root_path, "resources/model_checkpoint/goog_basic")

    # set up parameters
    params = {'attention_unit_size': 1024,
              'hidden_units': [2048, 2048, 1024, 512, 256],
              'n_classes': 3}

    # set up run config
    run_config = tf.estimator.RunConfig(save_summary_steps=10)

    # set the model
    classifier = tf.estimator.Estimator(
        model_fn=basic_predictor_fn,
        params=params,
        model_dir=model_path,
        config=run_config)

    classifier.train(lambda: time_irrelevant_train_fn(data_path, 'goog', 5), max_steps=100000)
