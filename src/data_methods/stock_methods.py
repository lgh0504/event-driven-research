from __future__ import print_function

import sqlite3
import time
from datetime import timedelta

import pandas as pd
from googlefinance.client import get_price_data


def _check_missed_data(df):
    """
    data from google finance sometime is not complete, so check and modify it myself
    :param df: the dataframe to be checked
    :return: a checked dataframe
    """
    i = 0
    while i < len(df) - 1:
        i += 1
        pre_time = df.iloc[i - 1]["Date"]
        cur_time = df.iloc[i]["Date"]
        delta_time = cur_time - pre_time

        # if the interval is not one minute
        if delta_time.seconds != 60 and pre_time.day == cur_time.day:
            # create a new entry
            new_time = pre_time + timedelta(minutes=1)
            new_open = df.iloc[i]["Open"]
            new_high = df.iloc[i]["High"]
            new_low = df.iloc[i]["Low"]
            new_close = df.iloc[i]["Close"]
            new_volume = df.iloc[i]["Volume"]
            line = pd.DataFrame({"Date": new_time, "Open": new_open, "High": new_high,
                                 "Low": new_low, "Close": new_close, "Volume": new_volume}, index=[i])

            # add the new entry to the dataframe
            df = pd.concat([df.iloc[:i], line, df.iloc[i:]]).reset_index(drop=True)
    return df


def get_good_stock_data(symbol, period, interval=60):
    """
    get a checked stock price data
    :param symbol: is the stock symbol like AAPL
    :param period: the time period of data
    :param interval: is the time interval
    :return: a formatted data frame
    """

    # get data from server
    param = {
        'q': symbol,  # Stock symbol (ex: "AAPL")
        'i': interval,  # Interval size in seconds ("86400" = 1 day intervals)
        'p': period  # Period
    }
    df = get_price_data(param)

    # reformat the index
    df.reset_index(inplace=True)
    df = df.rename(index=str, columns={"index": "Date"})

    # reformat time, from Beijing to USA east
    df['Date'] = df['Date'] + timedelta(hours=-12)

    # check the missed data
    df = _check_missed_data(df)

    # reorder data frame
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # change Date to string
    df['Date'] = df['Date'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:00'))

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
