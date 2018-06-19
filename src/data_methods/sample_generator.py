from __future__ import print_function
from datetime import datetime
from datetime import timedelta
from twitter_methods import TwitterDatabase
from stock_methods import StockDatabase


class SampleGenerator:
    # TODO: make the class more robust
    """
    SampleGenerator mainly responsible for generating training samples from database
    """

    def __init__(self, twitter_db_path, stock_db_path):
        """
        set up parameters and build connection to the database
        :param twitter_db_path: is the path to twitter database
        :param stock_db_path: is the path to stock database
        """

        # static parameters
        self.date_index = 0
        self.text_index = 1
        self.seconds_in_minute = 60
        self.minutes_in_day = 391
        self.market_open_time = datetime.strptime('9:30AM', '%I:%M%p').time()
        self.market_close_time = datetime.strptime('4:00PM', '%I:%M%p').time()
        self.stock_table_names = ['aal', 'aapl', 'adbe', 'adi', 'adp', 'adsk', 'akam', 'algn',
                                  'alxn', 'amat', 'amgn', 'amzn', 'atvi', 'avgo', 'bidu', 'biib',
                                  'bkng', 'bmrn', 'ca', 'celg', 'cern', 'chkp', 'chtr', 'cmcsa',
                                  'cost', 'csco', 'csx', 'ctas', 'ctrp', 'ctsh', 'ctxs', 'disca',
                                  'disck', 'dish', 'dltr', 'ea', 'ebay', 'esrx', 'expe', 'fast',
                                  'fb', 'fisv', 'fox', 'foxa', 'gild', 'goog', 'googl', 'has',
                                  'holx', 'hsic', 'idxx', 'ilmn', 'incy', 'intc', 'intu', 'isrg',
                                  'jbht', 'jd', 'khc', 'klac', 'lbtya', 'lbtyk', 'lila', 'lilak',
                                  'lrcx', 'mar', 'mat', 'mchp', 'mdlz', 'meli', 'mnst', 'msft',
                                  'mu', 'mxim', 'myl', 'nclh', 'nflx', 'ntes', 'nvda', 'orly',
                                  'payx', 'pcar', 'pypl', 'qcom', 'qrtea', 'regn', 'rost', 'sbux',
                                  'shpg', 'siri', 'stx', 'swks', 'symc', 'tmus', 'tsco', 'tsla', 'txn',
                                  'ulta', 'viab', 'vod', 'vrsk', 'vrtx', 'wba', 'wdc', 'wynn', 'xlnx', 'xray']

        # database connection
        self.twitter_db = TwitterDatabase(twitter_db_path)
        self.stock_db = StockDatabase(stock_db_path)

    def generate_tweets_list(self, query):
        # TODO: fix the weekend problem
        """
        generate a list of text buckets from database ( put the tweets to the corresponding buckets )
        :param query: is the SQL query to get data
        :return: a list of text buckets
        """

        # get query result from database
        query_result = self.twitter_db.query(query)

        # convert date, change the time outside of the trading time
        query_result = self._date_convert(query_result)

        # get start time and end time from all query result
        start_time, end_time = self._get_start_end_time(query_result)

        # convert index of the result
        query_result = self._index_convert(start_time, query_result)

        # return list of list of string
        return self._generate_text_bucket(start_time, end_time, query_result)

    def generate_stock_dict(self, start_time, end_time):
        # TODO: fix time problem
        stock_dict = {}
        for name in self.stock_table_names:
            query = "SELECT Open FROM " + name
            query_result = self.stock_db.query(query)
            query_result = map(lambda x: x[0], query_result)
            stock_dict[name] = query_result
        return stock_dict

    """ helper methods below """

    def _date_convert(self, query_result):
        """
        convert raw record's date to trading time date
        :param query_result: a list of (date, text)
        :return: data item is converted
        """

        # define some parameters
        open_hour = self.market_open_time.hour
        open_min = self.market_open_time.minute
        close_hour = self.market_close_time.hour
        close_min = self.market_close_time.minute

        # define a helper method
        def _time_convert(record):
            # get time of the record
            time = datetime.strptime(record[self.date_index], '%Y-%m-%d %H:%M:%S')

            # declare some boolean flag
            is_friday = time.weekday() == 5
            is_saturday = time.weekday() == 6
            is_sunday = time.weekday() == 7
            is_weekend = is_saturday or is_sunday
            too_early = self.market_open_time < time.time()
            too_late = time.time() < self.market_close_time
            not_trading_time = is_saturday or is_sunday or too_early or too_late

            # the time convert logic
            if not_trading_time:
                if is_weekend:
                    if is_saturday:
                        pass
                    if is_sunday:
                        pass
                if True:
                    pass

            if time.time() > self.market_close_time:
                time = time.replace(hour=close_hour, minute=close_min)
            time = time.replace(second=0)

            # generate new record
            new_record = [time, record[self.text_index]]

            return new_record

        # using the helper method
        return map(_time_convert, query_result)

    def _get_start_end_time(self, query_result):
        time_series = map(lambda record: record[self.date_index], query_result)
        start_time = min(time_series)
        end_time = max(time_series)
        return start_time, end_time

    def _index_convert(self, start_time, query_result):
        """
        convert record's time to index
        :param start_time: is the start time of all record
        :param query_result: is the query result
        :return: a new converted query result
        """

        def _date_to_key(record):
            time = record[self.date_index]
            delta_date = time - start_time
            delta_day = delta_date.days
            delta_min = delta_date.seconds / self.seconds_in_minute
            time_index = delta_day * self.minutes_in_day + delta_min
            new_record = [time_index, record[self.text_index]]
            return new_record

        return map(_date_to_key, query_result)

    def _generate_text_bucket(self, start_time, end_time, query_result):
        """
        generate text bucket at minute scale from query result
        :param start_time: is start time of all records
        :param end_time: is end time of all records
        :param query_result: is the query result
        :return: a list of list of text
        """
        time_duration = end_time - start_time
        bucket_number = time_duration.days * self.minutes_in_day + time_duration.seconds / self.seconds_in_minute + 1
        text_buckets = []
        for i in range(0, bucket_number):
            text_buckets.append([])

        for record in query_result:
            index = record[self.date_index]
            text = record[self.text_index]
            text_buckets[index].append(text)

        return text_buckets
