from os import path

from data_methods import StockDatabase
from data_methods import get_good_stock_data


def stock_data_pipeline(root_path, period):
    # set file path
    stock_symbol_path = path.join(root_path, "resources/nasdaq_symbol.txt")
    stock_db_path = path.join(root_path, "resources/new_nasdaq100_database.db")

    # connect to or create the database
    db = StockDatabase(stock_db_path)

    with open(stock_symbol_path) as f:
        stock_symbol_list = f.read().strip('\n').split(',')

        # the main loop, loop over stock
        for stock_symbol in stock_symbol_list:

            # get data (checked) from server
            df = get_good_stock_data(stock_symbol, period)

            # create the table
            table_name = stock_symbol.lower()
            db.create_table(table_name)

            # insert each row
            for index, row in df.iterrows():
                db.insert_row(row.tolist(), table_name)

            print("%s completed, %d records!" % (table_name, len(df)))


if __name__ == '__main__':
    # set up path
    root_path = "../.."

    stock_data_pipeline(root_path, "60d")
