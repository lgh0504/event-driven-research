import os
from os import path
from data_methods import stock_batch_read
from data_methods import StockDatabase

if __name__ == '__main__':

    # set up path
    root_path = os.environ['EVR_ROOT_PATH']
    print(root_path)

    # read all from xls file
    stock_xls_path = path.join(root_path, "resources/nasdaq_data/*.xls")
    start_time = '2018-04-23 09-30'
    end_time = '2018-04-27 16-00'
    print("start reading from file ...")
    stock_dict = stock_batch_read(stock_xls_path, start_time, end_time)
    check_list = map(lambda x: len(x), stock_dict.values())
    flag = True
    correct_number = 1955
    for number in check_list:
        if number != correct_number:
            flag &= False
    if flag:
        print("reading data success!")
        # insert df in to database
        stock_db_path = path.join(root_path, "resources/nasdaq100_database.db")
        db = StockDatabase(stock_db_path)

        # loop over all data frame
        for key, value in stock_dict.items():
            db.create_table(key)
            # insert all row
            for index, row in value.iterrows():
                db.insert_row(row.tolist(), key)
    else:
        print("reading data failed!")











