from __future__ import print_function
import pickle
import os
from os import path
import time


def generate_text_buckets():
    from data_methods import SampleGenerator

    # set up parameters
    twitter_db_path = path.join(root_path, "resources/twitter_database.db")
    stock_db_path = path.join(root_path, "resources/nasdaq100_database.db")
    query = "SELECT Date,text from Tweets WHERE followers_count > 10000 " \
            "AND DATETIME(Date) >= '2018-04-23 09:00:00' AND DATETIME(Date) <= '2018-04-27 16:00:01'"

    # generate text buckets and dump the result
    sample_generator = SampleGenerator(twitter_db_path, stock_db_path)
    text_buckets = sample_generator.generate_tweets_list(query)
    pickle.dump(text_buckets, open(path.join(root_path, "resources/training_data/text_buckets.pickle"), "wb"))


def generate_stock_tweet_dict():

    # build look up table
    table_path = path.join(root_path, "resources/lookup_table.txt")
    lookup_table = {}
    with open(table_path, "rb") as f:
        for line in f:
            line = line[0:-1]
            key = line.split(":")[0]
            values = line.split(":")[1]
            values = values.split(",")
            lookup_table[key] = values

    print (lookup_table)
    stock_tweet_dict = {}
    stock_table_names = ['aal', 'aapl', 'adbe', 'adi', 'adp', 'adsk', 'akam', 'algn',
                         'alxn', 'amat', 'amgn', 'amzn', 'atvi', 'avgo', 'bidu', 'biib',
                         'bkng', 'bmrn', 'ca', 'celg', 'cern', 'chkp', 'chtr', 'cmcsa',
                         'cost', 'csco', 'csx', 'ctas', 'ctrp', 'ctsh', 'ctxs', 'disca',
                         'disck', 'dish', 'dltr', 'ea', 'ebay', 'esrx', 'expe', 'fast',
                         'fb', 'fisv', 'fox', 'foxa', 'gild', 'goog', 'googl', 'has',
                         'holx', 'hsic', 'idxx', 'ilmn', 'incy', 'intc', 'intu', 'isrg',
                         'jbht', 'jd', 'khc', 'klac', 'lbtya', 'lbtyk', 'lila', 'lilak',
                         'lrcx', 'mar', 'mat', 'mchp', 'mdlz', 'meli', 'mnst', 'msft',
                         'mu', 'mxim', 'myl', 'nclh', 'nflx', 'ntes', 'nvda', 'orly',
                         'payx', 'pcar', 'pypl', 'qcom', 'qrtea', 'regn', 'rost', 'sbux',
                         'shpg', 'siri', 'stx', 'swks', 'symc', 'tmus', 'tsco', 'tsla', 'txn',
                         'ulta', 'viab', 'vod', 'vrsk', 'vrtx', 'wba', 'wdc', 'wynn', 'xlnx', 'xray']

    for name in stock_table_names:
        stock_tweet_dict[name] = []
        for i in range(0, 1955):
            stock_tweet_dict[name].append([])

    # load data
    data_path = path.join(root_path, "resources/training_data/text_buckets.pickle")
    text_buckets = pickle.load(open(data_path, "rb"))

    # range
    sel = ['goog', 'msft', 'amzn', 'intc', 'aapl', 'nflx', 'ebay', 'fb']
    index = -1
    for bucket in text_buckets:
        index += 1
        for text in bucket:
            for name in stock_tweet_dict:
                for value in lookup_table[name]:
                    if value.lower() in text.lower():
                        stock_tweet_dict[name][index].append(text)

    # dump data
    pickle.dump(stock_tweet_dict, open(path.join(root_path, "resources/training_data/stock_tweet_dict.pickle"), "wb"))


def generate_tweet_vector_buckets():
    from data_methods import TextEmbedding

    # set up parameters
    model_path = path.join(root_path, "resources/GoogleNews-vectors-negative300.bin")
    data_path = path.join(root_path, "resources/training_data/text_buckets.pickle")

    # load model
    text_embedder = TextEmbedding(model_path)

    # load data
    print("load data...")
    start_time = time.time()
    text_buckets = pickle.load(open(data_path, "rb"))
    end_time = time.time()
    print("load data completed using time %d second" % (end_time - start_time))

    # embedding
    vector_buckets = []
    count = 0
    day = 0
    day_number = 391
    for text_list in text_buckets:
        count += 1

        vector_buckets.append(map(TextEmbedding.vectors_mean, map(text_embedder.text_embedding, text_list)))
        if count % day_number == 0:
            day += 1
            print("%d day completed" % day)

    # dump the result
    print("dump data...")
    start_time = time.time()
    pickle.dump(vector_buckets, open(path.join(root_path, "resources/training_data/vector_buckets.pickle"), "wb"))
    end_time = time.time()
    print("dump data completed using time %d second" % (end_time - start_time))


def generate_stock_dict():
    from data_methods import SampleGenerator

    twitter_db_path = path.join(root_path, "resources/twitter_database.db")
    stock_db_path = path.join(root_path, "resources/nasdaq100_database.db")
    stock_dict = SampleGenerator(twitter_db_path, stock_db_path).generate_stock_dict(0, 0)
    pickle.dump(stock_dict, open(path.join(root_path, "resources/training_data/stock_dict.pickle"), "wb"))


if __name__ == "__main__":
    root_path = os.environ['EDR_ROOT_PATH']
    generate_stock_tweet_dict()
