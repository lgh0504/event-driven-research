import json
import sys
import re
import tensorflow as tf
from os import path
from os import environ

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
model_path = "hdfs://cluster-test-m:8020/tmp"  # change path to DFS

# set distributed parameters
# master: 10.162.0.4    35.203.57.22
# worker0: 10.162.0.6   35.203.99.177
# worker1 : 10.162.0.7  35.203.98.108
config = {"cluster": {'ps': ['10.162.0.4:2222'],
                      'chief': ['10.162.0.6:2222'],
                      'worker': ['10.162.0.7:2222']}}

command_argv = sys.argv[1]
pattern = "(\w{1,10})_(\d{1,10})"
my_type = re.search(pattern, command_argv).group(1)
my_index = int(re.search(pattern, command_argv).group(2))
config_dict = {'type': my_type, 'index': my_index}
config["task"] = config_dict
environ['TF_CONFIG'] = json.dumps(config)

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
    config=tf.estimator.RunConfig().replace(save_summary_steps=10))

train_spec = tf.estimator.TrainSpec(lambda: train_input_fn(100, max_time, batch_size), max_steps=100)
eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(10, max_time, batch_size))
tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
