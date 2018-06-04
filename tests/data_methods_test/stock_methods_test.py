from os import path
from data_methods import stock_batch_read
from data_methods import StockDatabase

current_path = path.dirname(path.abspath(__file__))
root_path = path.dirname(path.dirname(current_path))


def stock_batch_read_test():
    # read all from xls file
    stock_xls_path = path.join(root_path, "resources/nasdaq_data/*.xls")
    start_time = '2018-04-23 09-30'
    end_time = '2018-04-27 16-00'
    stock_dict = stock_batch_read(stock_xls_path, start_time, end_time)
    print(map(lambda x: len(x), stock_dict.values()))


def stock_database_test():
    pass


if __name__ == "__main__":
    stock_batch_read_test()
