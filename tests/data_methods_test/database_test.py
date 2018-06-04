from os import path

current_path = path.dirname(path.abspath(__file__))
parent_path = path.dirname(current_path)

from data_methods import *


db_path = path.join(parent_path, 'resources/twitter_database.db')
db = TwitterDatabase(db_path)
print(db.query("SELECT MIN(Date), MAX(Date), COUNT(*) FROM Tweets"))

# db_path = path.join(parent_path, 'resources/nasdaq100_database.db')
# stock_xls_path = path.join(parent_path, "resources/nasdaq_data/*.xls")
# db = StockDatabase(db_path)
# stock_names = stock_names(stock_xls_path)
# for name in stock_names:
#     print(db.query("SELECT COUNT(*) FROM " + name))
