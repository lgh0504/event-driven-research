from __future__ import print_function
import pickle
import os
from os import path
import time


# all data together in the bucket
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


# rule based filter, generate 107 buckets based on text buckets
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

    print(lookup_table)
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

    # load data
    data_path = path.join(root_path, "resources/training_data/text_buckets.pickle")
    text_buckets = pickle.load(open(data_path, "rb"))

    # create the dict
    for name in stock_table_names:
        stock_tweet_dict[name] = []
        for i in range(0, len(text_buckets)):
            stock_tweet_dict[name].append([])

    # range
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


# all data embedding
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


# only embedding top8 stock
def generate_top8_vector_buckets():
    from data_methods import TextEmbedding

    # set up parameters
    model_path = path.join(root_path, "resources/GoogleNews-vectors-negative300.bin")
    data_path = path.join(root_path, "resources/training_data/stock_tweet_dict.pickle")

    # load model
    text_embedder = TextEmbedding(model_path)

    # load data
    print("load data...")
    start_time = time.time()
    stock_tweet_dict = pickle.load(open(data_path, "rb"))
    end_time = time.time()
    print("load data completed using time %d second" % (end_time - start_time))

    # embedding
    sel = ['goog', 'msft', 'amzn', 'intc', 'aapl', 'nflx', 'ebay', 'fb']
    stock_vector_dict = {}
    for key in sel:
        vector_buckets = []
        stock_vector_dict[key] = vector_buckets
        count = 0
        day = 0
        day_number = 391
        for text_list in stock_tweet_dict[key]:
            count += 1
            vector_buckets.append(map(TextEmbedding.vectors_mean, map(text_embedder.text_embedding, text_list)))
            if count % day_number == 0:
                day += 1
                print("%d day completed" % day)
        print("stock %s completed!" % key)

    # dump the result
    print("dump data...")
    start_time = time.time()
    pickle.dump(stock_vector_dict, open(path.join(root_path, "resources/training_data/stock_vector_dict.pickle"), "wb"))
    end_time = time.time()
    print("dump data completed using time %d second" % (end_time - start_time))


# generate stock price dict (open price)
def generate_stock_dict():
    from data_methods import SampleGenerator

    twitter_db_path = path.join(root_path, "resources/twitter_database.db")
    stock_db_path = path.join(root_path, "resources/nasdaq100_database.db")
    stock_dict = SampleGenerator(twitter_db_path, stock_db_path).generate_stock_dict(0, 0)
    pickle.dump(stock_dict, open(path.join(root_path, "resources/training_data/stock_dict.pickle"), "wb"))


# generate time-irrelevant training sample
def generate_time_irrelevant_sample(time_interval):
    # set up parameters
    import sqlite3
    tweet_path = path.join(root_path, "resources/training_data/stock_vector_dict.pickle")
    stock_path = path.join(root_path, "resources/training_data/stock_dict.pickle")
    stock_db_path = path.join(root_path, "resources/nasdaq100_database.db")
    # load data
    print("load data...")
    start_time = time.time()
    stock_vector_dict = pickle.load(open(tweet_path, "rb"))
    stock_dict = pickle.load(open(stock_path, "rb"))
    end_time = time.time()
    print("load data completed using time %d second" % (end_time - start_time))

    # generate some data dynamically
    UP = [1, 0, 0]
    PRESERVE = [0, 1, 0]
    DOWN = [0, 0, 1]
    sel = ['goog', 'msft', 'amzn', 'intc', 'aapl', 'nflx', 'ebay', 'fb']
    loop_length = len(stock_dict[sel[0]]) - time_interval

    def _boundary_select(time_interval, stock_db_path):
        boundary_list = {}
        courser = sqlite3.connect(stock_db_path).cursor()

        # get boundary of each stock
        for key in sel:
            # get price info from database
            query_result = courser.execute("SELECT Open FROM " + key)
            price_list = map(lambda x: x[0], query_result)
            loop_length = len(price_list) - time_interval
            price_delta = []

            # get delta between prices
            for i in range(0, loop_length):
                price_delta.append((price_list[i + time_interval] - price_list[i]) / price_list[i])

            # sort the price list and get static info
            price_delta.sort()
            price_delta = map(lambda x: x * 100, price_delta)  # times 100 to get % data
            down = price_delta[len(price_delta) / 3]
            up = price_delta[len(price_delta) * 2 / 3]
            boundary_list[key] = (down, up, price_delta)

        # get overall threshold
        global_down_boundary = sum(map(lambda x: x[0], boundary_list.values())) / len(sel)
        global_up_boundary = sum(map(lambda x: x[1], boundary_list.values())) / len(sel)

        # generate statical report
        # def down_count(my_list, down):
        #     count = 0
        #     for num in my_list:
        #         if num < down:
        #             count += 1
        #     return count
        #
        # def up_count(my_list, up):
        #     count = 0
        #     for num in my_list:
        #         if num > up:
        #             count += 1
        #     return count
        # for key in sel:
        #     my_list = boundary_list[key][2]
        #     print(key, down_count(my_list, global_up_boundary) * 100 / float(len(my_list)),
        #           up_count(my_list, global_down_boundary) * 100 / float(len(my_list)))

        return global_down_boundary, global_up_boundary

    DOWN_THRESHOLD, UP_THRESHOLD = _boundary_select(time_interval, stock_db_path)

    # the main loop to generate data
    training_dict = {}
    for key in sel:
        count = 0
        for i in range(0, loop_length):
            # generate text bucket in this time interval
            text_bucket = []
            for j in range(0, time_interval):
                text_bucket += stock_vector_dict[key][i+j]
            # decide whether include it
            if len(text_bucket) > 0:
                count += 1
                price_delta = (stock_dict[key][i + time_interval] - stock_dict[key][i]) * 100 / stock_dict[key][i]
                if price_delta < DOWN_THRESHOLD:
                    label = DOWN
                elif price_delta > UP_THRESHOLD:
                    label = UP
                else:
                    label = PRESERVE
                if key not in training_dict:
                    training_dict[key] = [(text_bucket, label)]
                else:
                    training_dict[key].append((text_bucket, label))

        print("stock %s completed" % key)
        print("total %d samples" % count)

    # dump the result
    print("dump data...")
    start_time = time.time()
    pickle.dump(training_dict, open(path.join(root_path, "resources/training_data/training_dict.pickle"), "wb"))
    end_time = time.time()
    print("dump data completed using time %d second" % (end_time - start_time))


if __name__ == "__main__":
    root_path = os.environ['EDR_ROOT_PATH']
    generate_time_irrelevant_sample(10)
