import pickle
import time
import tensorflow as tf


def _padding_work(list_list, labels):

    # static work
    zero_vec = [0] * 300
    length_list = map(lambda x: len(x), list_list)
    length_list.sort()
    drop_num = len(length_list) / 20
    max_len = length_list[-drop_num]
    print max_len

    # filter
    padding_list = []
    padding_labels = []
    index = -1
    for the_list in list_list:
        index += 1
        if len(the_list) <= max_len:
            for i in range(0, max_len - len(the_list)):
                the_list.append(zero_vec)
            padding_list.append(the_list)
            padding_labels.append(labels[index])
    return padding_list, padding_labels


def time_irrelevant_train_fn(data_path, stock_symbol, batch_size):
    # load data
    print("load data...")
    start_time = time.time()
    train_dict = pickle.load(open(data_path, "rb"))
    train_sample = train_dict[stock_symbol]
    end_time = time.time()
    print("load data completed using time %d second" % (end_time - start_time))

    # generate dataset
    samples = map(lambda x: x[0], train_sample)
    labels = map(lambda x: x[1], train_sample)

    # do padding work
    samples, labels= _padding_work(samples, labels)
    print("padding completed!")

    # generate training data
    features = {'news_bucket': samples}
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.repeat(10).batch(batch_size)
