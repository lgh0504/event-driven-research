from __future__ import print_function
import glob
import re
import time
import sqlite3
from datetime import datetime
from datetime import timedelta
import pandas as pd


def stock_batch_read(resource_path, start_time, end_time):
    """
    read all xls file in the resource path
    :param resource_path: path to the directory containing xls files
    :param start_time: start time of stock price
    :param end_time: end time of stock price
    :return: a dictionary, key is stock name, value is the a data frame of the xls file
    """
    # read all xls files
    stock_files = glob.glob(resource_path)
    pattern = "_(\w{1,10})\.xls"

    # collect the stock dictionary
    stock_dict = {}
    for filename in stock_files:
        stock_symbol = re.search(pattern, filename).group(1)
        df = pd.read_excel(filename)

        # get rid of time outside boundary
        df = df[(df['Date'] >= start_time) & (df['Date'] <= end_time)]

        # convert data format
        df['Date'] = df['Date'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:00',
                                                              time.strptime(x, '%Y-%m-%d %H-%M')))
        # check the missed data
        df = _check_missed_data(df)

        # generate the stock : data frame dict
        stock_dict[stock_symbol] = df

    # return the result
    return stock_dict


def _check_missed_data(df):
    for i in range(0, len(df)):
        if i == 0:
            pass
        else:
            pre_time = datetime.strptime(df.iloc[i - 1]["Date"], '%Y-%m-%d %H:%M:%S')
            cur_time = datetime.strptime(df.iloc[i]["Date"], '%Y-%m-%d %H:%M:%S')
            delta_time = cur_time - pre_time
            if delta_time.seconds != 60 and pre_time.day == cur_time.day:
                new_time = (pre_time + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
                new_open = df.iloc[i]["Open"]
                new_high = df.iloc[i]["High"]
                new_low = df.iloc[i]["Low"]
                new_close = df.iloc[i]["Close"]
                new_volume = df.iloc[i]["Volume"]
                line = pd.DataFrame({"Date": new_time, "Open": new_open, "High": new_high,
                                     "Low": new_low, "Close": new_close, "Volume": new_volume}, index=[i])
                df = pd.concat([df.ix[:i - 1], line, df.ix[i:]]).reset_index(drop=True)
    return df


class StockDatabase:
    """
    Methods related to stock database
    """

    def __init__(self, db_path):
        """
        create the link to the database
        :param db_path: is the path to the stock database
        """
        self.row_format = " (Date DATETIME, Open FLOAT, High FLOAT, Low FLOAT, Close FLOAT, Volume FLOAT)"
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.insert_count = 0

    def __del__(self):
        self.conn.commit()
        self.conn.close()
        print("Nasdaq Database closed")

    def create_table(self, table_name):
        """
        create a table
        :param table_name: is the name of the table
        :return: nothing
        """
        create_table_sql = "CREATE TABLE IF NOT EXISTS " + table_name + self.row_format
        self.cursor.execute(create_table_sql)

    def insert_row(self, row, table_name, batch_mode=True):
        """
        insert a list as a row into the stock database
        :param row: list of a record
        :param table_name: name of table to insert into
        :param batch_mode: performance choice
        :return: nothing
        """
        row_format = '(' + ','.join(['?'] * len(row)) + ')'
        self.cursor.execute('INSERT INTO %s VALUES %s' % (table_name, row_format), row)

        if not batch_mode:
            self.conn.commit()
            return

        self.insert_count += 1
        if self.insert_count % 10000 == 0:
            self.conn.commit()
            self.insert_count = 0
            print("Commit 10000 inserts...")

    def query(self, query):
        """
        execute a query
        :param query: a SQL format string
        :return: a list of row(tuple)
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()
