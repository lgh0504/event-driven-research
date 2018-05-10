import sys
import tensorflow as tf
from tensorflow.contrib.distribute import MirroredStrategy
from os import path
current_path = path.dirname(path.abspath(__file__))
root_path = path.dirname(current_path)
sys.path.append(root_path)
from models.model_fn import hybrid_fn
from models.input_fn import train_input_fn
from models.input_fn import eval_input_fn

# set hyper parameters
batch_size = 5
max_time = 15
state_size = 4
keep_rate = 0.9
label_size = 1

# set model dir
model_path = "./"  # change path to DFS

# set up the config
distribution = MirroredStrategy(num_gpus=2)
run_config = tf.estimator.RunConfig(train_distribute=distribution, save_summary_steps=10)

# set the model
classifier = tf.estimator.Estimator(
    model_fn=hybrid_fn.complex_model,
    params={
        'batch_size': batch_size,
        'state_size': state_size,
        'truncated_backprop_length': max_time,
        'keep_rate': keep_rate,
        'label_size': label_size
    },
    model_dir=model_path,
    config=run_config)

train_spec = tf.estimator.TrainSpec(lambda: train_input_fn(100, max_time, batch_size), max_steps=100)
eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(10, max_time, batch_size))
tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
