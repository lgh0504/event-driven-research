from __future__ import print_function
import numpy as np
from datetime import datetime
from datetime import timedelta
from twitter_methods import TwitterDatabase
from stock_methods import StockDatabase
from embedding_methods import 

class SampleGenerator:
    """ SampleGenerator mainly responsible for generating training samples from database """

    def __init__(self, twitter_db_path, stock_db_path, lookup_table_path,
                 start_date, start_index, end_date, end_index,
                 stock_pool=('goog', 'msft', 'amzn', 'intc', 'aapl', 'nflx', 'ebay', 'fb')):
        """
        1. set up parameters and build connection to the database
        2. generate the key data structure
        :param twitter_db_path: is the path to twitter database
        :param stock_db_path: is the path to stock database
        :param start_date: is 'yyyy-mm-dd'
        :param start_index: is from 0 to 391
        :param end_date: is 'yyyy-mm-dd'
        :param end_index: is from 0 to 391
        """

        # static parameters
        self.date_index = 0
        self.text_index = 1
        self.seconds_in_minute = 60
        self.minutes_in_day = 391
        self.market_open_time = datetime.strptime('9:30AM', '%I:%M%p').time()
        self.market_close_time = datetime.strptime('4:00PM', '%I:%M%p').time()
        self.stock_pool = stock_pool
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
        self.lookup_table = self.build_lookup_table(lookup_table_path)

        # generate twitter list [(date, index, [text])]
        # date is datetime, index is int, text is string
        self.twitter_list = self._generate_tweets_list(self._generate_twitter_query(
            start_date, start_index, end_date, end_index
        ))

        # generate stock dict {symbol: [(date, index, price)]}
        # date is datetime, index is int, price is float
        self.stock_dict = self._generate_stock_dict(self._generate_stock_query(
            start_date, start_index, end_date, end_index
        ))

        # filter twitter_list by joining two data structure
        self._join(self.twitter_list, self.stock_dict['aapl'])

    """ APIs for generating training samples """

    # TODO: design more reasonable APIs, now training sample generator has problem
    def get_serial_text_sample(self, stock_symbol, time_interval):
        """
        return well-formatted training data
        :param stock_symbol: is the stock symbol data to get
        :param time_interval: is the training time
        :return:
        """
        serial_text_list = []
        filtered_twitter_list = self.filter_by_keywords(stock_symbol)
        for i in range(0, len(self.twitter_list) - time_interval + 1):
            serial_text_list.append(filtered_twitter_list[i:i + time_interval])
        return serial_text_list

    def get_serial_stock_sample(self, symbol_list, time_interval):
        stock_list = []
        for symbol in symbol_list:
            stock_list.append(map(lambda x: x[2], self.stock_dict[symbol]))
        stock_list = np.matrix(stock_list).transpose().tolist()

        serial_stock_list = []
        for i in range(0, len(self.twitter_list) - time_interval + 1):
            serial_stock_list.append(stock_list[i:i + time_interval])
        return serial_stock_list

    def get_stock_and_compliment(self, stock_symbol, time_interval):
        compliment_list = [e for e in self.stock_table_names if e != stock_symbol]
        return (self.get_serial_stock_sample([stock_symbol], time_interval),
                self.get_serial_stock_sample(compliment_list, time_interval))

    def new_get_serial_text_sample(){

    }

    """ first stage helper methods """

    @staticmethod
    def build_lookup_table(table_path):
        lookup_table = {}
        with open(table_path, "rb") as f:
            for line in f:
                line = line[0:-1]
                key = line.split(":")[0]
                values = line.split(":")[1]
                values = values.split(",")
                lookup_table[key] = values
        return lookup_table

    def filter_by_keywords(self, stock_symbol):
        def _filter(tweets):
            filtered_tweets = []
            for tweet in tweets:
                for value in self.lookup_table[stock_symbol]:
                    if value.lower() in tweet.lower():
                        filtered_tweets.append(tweet)
            return filtered_tweets

        return map(_filter, map(lambda x: x[2], self.twitter_list))

    def _generate_twitter_query(self, start_date, start_index, end_date, end_index):
        # very complex logic to generate start time
        if start_index != 0:
            start_time = "\'" + start_date + " " + self._generate_time(start_index - 1) + "\'"
        else:
            # TODO: simple logic here now, maybe considering weekend
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            start_date += timedelta(days=-1)
            start_date = start_date.strftime("%Y-%m-%d")
            start_time = "\'" + start_date + " 16:00\'"

        end_time = "\'" + end_date + " " + self._generate_time(end_index) + "\'"
        query = " AND DATETIME(Date) >= " + start_time + " AND DATETIME(Date) < " + end_time  # >= and < is tricky
        return query

    def _generate_stock_query(self, start_date, start_index, end_date, end_index):

        start_time = "\'" + start_date + " " + self._generate_time(start_index) + "\'"
        end_time = "\'" + end_date + " " + self._generate_time(end_index) + "\'"
        query = " WHERE DATETIME(Date) >= " + start_time + " AND DATETIME(Date) <= " + end_time  # >= and <= is tricky
        return query

    def _generate_tweets_list(self, partial_query):

        # get query result from database
        query = "SELECT Date,text from Tweets WHERE followers_count > 10000" + partial_query
        query_result = self.twitter_db.query(query)

        # convert date format to date + index, return [(date, index, text)]
        # date is datetime, index is int, text is string
        query_result = self._twitter_date_convert(query_result)

        # merge [(date, index, text)] to [(date, index, [text])]
        query_result = self._merge_text(query_result)

        # sort the data
        zero = datetime.min.time()
        query_result = sorted(query_result, key=lambda x: datetime.combine(x[0], zero) + timedelta(minutes=x[1]))

        return query_result

    def _generate_stock_dict(self, partial_query):

        # generate the stock dict
        stock_dict = {}

        # loop over all stock symbol
        for name in self.stock_table_names:
            # generate the query, get ((Date, Price)) set from database
            query = "SELECT DISTINCT Date,Open FROM " + name + partial_query
            query_result = self.stock_db.query(query)

            # convert data to [(Date, index, Price)], while the Date is datetime type
            query_result = self._stock_date_convert(query_result)

            # sort the data
            zero = datetime.min.time()
            query_result = sorted(query_result, key=lambda x: datetime.combine(x[0], zero) + timedelta(minutes=x[1]))

            stock_dict[name] = query_result

        return stock_dict

    def _join(self, twitter_list, stock_list):

        joined_twitter_list = []

        # define a hash function
        def _my_hash(date, index):
            hash_key = date.strftime('%Y-%m-%d') + str(index)
            return hash_key

        # build hash table
        hash_dict = {}
        for record in stock_list:
            hash_dict[_my_hash(record[0], record[1])] = 0

        # join by nested hashing
        for record in twitter_list:
            if _my_hash(record[0], record[1]) in hash_dict:
                joined_twitter_list.append(record)

        self.twitter_list = joined_twitter_list

    """ second stage helper methods """

    @staticmethod
    def _generate_time(index):
        open_time = datetime.strptime('9:30AM', '%I:%M%p')
        time_generated = open_time + timedelta(minutes=index)
        time_generated = time_generated.time()
        return time_generated.strftime('%H:%M:%S')

    @staticmethod
    def _merge_text(query_result):

        # make an empty dict for hashing
        hash_dict = {}

        # define a hash function
        def _my_hash(date, index):
            hash_key = date.strftime('%Y-%m-%d') + str(index)
            return hash_key

        # hashing
        for record in query_result:
            key = _my_hash(record[0], record[1])
            if key not in hash_dict:
                hash_dict[key] = [record]
            else:
                hash_dict[key].append(record)

        # merge the dict
        query_result = []
        for record_list in hash_dict.values():
            date = record_list[0][0]
            index = record_list[0][1]
            text_list = []
            for record in record_list:
                text_list.append(record[2])
            key_tuple = (date, index, text_list)
            query_result.append(key_tuple)
        return query_result

    def _twitter_date_convert(self, query_result):
        """
        convert raw record's date to trading time date
        and make the indexing work easier
        :param query_result: a list of (date, text)
        :return: data item is converted
        """

        # define a helper method
        def _time_convert(record):
            # get time of the record
            time = datetime.strptime(record[self.date_index], '%Y-%m-%d %H:%M:%S')
            trading_date = time.date()
            trading_time = time.time()
            trading_index = 0

            # declare some boolean flag
            is_friday = time.weekday() == 5
            is_saturday = time.weekday() == 6
            is_sunday = time.weekday() == 7
            too_early = trading_time < self.market_open_time  # < and >= make it easier for logic below and it's right
            too_late = trading_time >= self.market_close_time

            # the time convert logic
            # for time not in trading time, modify the trading date
            # for time in trading time, modify the trading index
            if is_saturday:
                trading_date += timedelta(days=2)
            elif is_sunday:
                trading_date += timedelta(days=1)
            elif is_friday and too_late:
                trading_date += timedelta(days=3)
            elif too_late:
                trading_date += timedelta(days=1)
            elif too_early:
                pass
            else:
                open_time = self.market_open_time
                index_time = time + timedelta(minutes=1)
                index_time = index_time.time()
                trading_index = (index_time.hour - open_time.hour) * 60 + (index_time.minute - open_time.minute)

            # generate new record
            new_record = (trading_date, trading_index, record[self.text_index])

            return new_record

        # using the helper method
        query_result = map(_time_convert, query_result)
        return query_result

    def _stock_date_convert(self, query_result):

        open_time = self.market_open_time

        def _time_convert(record):
            # get time of the record
            time = datetime.strptime(record[self.date_index], '%Y-%m-%d %H:%M:%S')
            trading_date = time.date()
            trading_time = time.time()
            trading_index = (trading_time.hour - open_time.hour) * 60 + (trading_time.minute - open_time.minute)

            # generate new record
            new_record = (trading_date, trading_index, record[self.text_index])

            return new_record

        # using the helper method
        query_result = map(_time_convert, query_result)
        return query_result

if __name__ == "__main__":
    sample = SampleGenerator(
        '../../resources/twitter_database.db',
        '../../resources/nasdaq100_database.db',
        '../../resources/filter_table.txt',
        '2018-04-23',
        0,
        '2018-04-30',
        391,
    )

