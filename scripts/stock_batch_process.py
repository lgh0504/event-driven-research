from os import path
import data_methods as dm

if __name__ == '__main__':
    current_path = path.dirname(path.abspath(__file__))
    root_path = path.dirname(current_path)

    # read all from xls file
    stock_xls_path = path.join(root_path, "resources/NASDAQ_data/*.xls")
    start_time = '2018-04-23 09-30'
    end_time = '2018-04-27 16-00'
    stock_dict = dm.stock_batch_read(stock_xls_path, start_time, end_time)

    # insert df in to database
    stock_db_path = path.join(root_path, "resources/nasdaq100_database.db")
    db = dm.StockDatabase(stock_db_path)

    # loop over all data frame
    for key, value in stock_dict.items():
        db.create_table(key)
        # insert all row
        for index, row in value.iterrows():
            db.insert_row(row.tolist(), key)









