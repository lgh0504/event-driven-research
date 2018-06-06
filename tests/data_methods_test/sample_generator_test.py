from data_methods import SampleGenerator

""" test methods below """


def generate_tweets_list_test(twitter_database_path, stock_database_path):
    query = "SELECT Date,text from Tweets WHERE followers_count > 100000 \
               AND DATETIME(Date) >=  '2018-04-23 09:10:00' AND DATETIME(Date) <= '2018-04-23 09:40:00'"
    text_bucket = SampleGenerator(twitter_database_path, stock_database_path).generate_tweets_list(query)
    print(map(lambda x: len(x), text_bucket))


def generate_stock_dict_test(twitter_database_path, stock_database_path):
    stock_dict = SampleGenerator(twitter_database_path, stock_database_path).generate_stock_dict(0, 0)
    print(len(stock_dict['aal']))


if __name__ == "__main__":
    from os import path

    current_path = path.dirname(path.abspath(__file__))
    root_path = path.dirname(path.dirname(current_path))
    twitter_db_path = path.join(root_path, "resources/twitter_database.db")
    stock_db_path = path.join(root_path, "resources/nasdaq100_database.db")
    generate_stock_dict_test(twitter_db_path, stock_db_path)
