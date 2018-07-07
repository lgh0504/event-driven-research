from __future__ import print_function
from data_methods.twitter_methods import TwitterDatabase
from data_methods.stock_methods import StockDatabase


def twitter_database_test(db_path):
    db = TwitterDatabase(db_path)
    query0 = "SELECT Date,text from Tweets WHERE followers_count > 10000 " \
             "AND DATETIME(Date) >= '2018-04-23 09:00:00' AND DATETIME(Date) <= '2018-04-27 16:00:01'"
    query1 = "SELECT MIN(Date), MAX(Date) FROM Tweets"
    print(db.query(query1))


def stock_database_test(db_path):
    db = StockDatabase(db_path)
    query0 = "SELECT COUNT(*) FROM (SELECT DISTINCT Date FROM aapl)"
    query1 = "SELECT MIN(Date), MAX(Date) FROM aapl"
    print(db.query(query1))


if __name__ == "__main__":
    from os import path

    # set up path
    current_path = path.dirname(path.abspath(__file__))
    parent_path = path.dirname(path.dirname(current_path))
    twitter_db_path = path.join(parent_path, 'resources/big_twitter_database.db')
    stock_db_path = path.join(parent_path, 'resources/new_nasdaq100_database.db')

    # test
    twitter_database_test(twitter_db_path)
    stock_database_test(stock_db_path)
