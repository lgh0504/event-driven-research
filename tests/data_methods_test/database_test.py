from data_methods.twitter_methods import TwitterDatabase


def twitter_database_test(db_path):
    db = TwitterDatabase(db_path)
    query0 = "SELECT Date,text from Tweets WHERE followers_count > 10000 " \
             "AND DATETIME(Date) >= '2018-04-23 09:00:00' AND DATETIME(Date) <= '2018-04-27 16:00:01'"
    query1 = "SELECT MIN(Date), MAX(Date), COUNT(*) FROM Tweets"
    print(len(db.query(query0)))


# db_path = path.join(parent_path, 'resources/nasdaq100_database.db')
# stock_xls_path = path.join(parent_path, "resources/nasdaq_data/*.xls")
# db = StockDatabase(db_path)
# stock_names = stock_names(stock_xls_path)
# for name in stock_names:
#     print(db.query("SELECT COUNT(*) FROM " + name))

if __name__ == "__main__":
    from os import path

    # set up path
    current_path = path.dirname(path.abspath(__file__))
    parent_path = path.dirname(path.dirname(current_path))

    # test
    db_path = path.join(parent_path, 'resources/twitter_database.db')
    twitter_database_test(db_path)
