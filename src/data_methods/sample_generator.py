from __future__ import print_function
import numpy as np
from datetime import datetime
from datetime import timedelta
from twitter_methods import TwitterDatabase
from stock_methods import StockDatabase
from embedding_methods import TextEmbedding
import math

class SampleGenerator:
    """ SampleGenerator mainly responsible for generating training samples from database """

    def __init__(self, twitter_db_path, stock_db_path, lookup_table_path, general_words_path,
                 model_path,start_date, start_index, end_date, end_index,
                 stock_pool=('goog', 'msft', 'amzn', 'intc', 'aapl', 'nflx', 'ebay', 'fb')):
        """
        1. set up parameters and build connection to the database
        2. generate the key data structure
        :param twitter_db_path: is the path to twitter database
        :param stock_db_path: is the path to stock database
        :param lookup_table_path: lookup table specifies specific filters of one stock needs 
        :param general_words_path: general filters of one stock
        :param model_path: word embedding
        :param start_date: is 'yyyy-mm-dd'
        :param start_index: is from 0 to 391
        :param end_date: is 'yyyy-mm-dd'
        :param end_index: is from 0 to 391
        """

        # static parameters
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

        # database connection, lookup table building, word2vev model loading
        self.twitter_db = TwitterDatabase(twitter_db_path)
        self.stock_db = StockDatabase(stock_db_path)
        self.lookup_table = self._build_lookup_table(lookup_table_path)
        self.general_words = self._read_general_words(general_words_path)
        # DEBUG
        self.model = TextEmbedding(model_path)
        # self.model = None
        # self.date_index = 0
        # self.text_index = 0
        #DEBUG

        # generate twitter list [(date, index, [(text, count)])]
        # date is datetime, index is int, text is string, count is int
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

    def get_serial_text_sample(self, stock_symbol, time_interval):
        # TODO: design more reasonable APIs, now training sample generator has problem
        # POLICY: ver1. if not full 391 entries for one single day, we drop the whole day
        # POLICY: ver2. allow at most one point data loss 
        """

        :param stock_symbol:
        :param time_interval:
        :return: [[[vector,...](one window),...](sliding window),...]
        """
        training_samples = []
        list = self._embedding(self._get_stock_tweets(stock_symbol)) #[(date, index, vector)]
        #[[li[0]]+li[i:i+4] for i in range(1,100-4)]
        # analyze sample pools 
        sample_bitmap = {}
        for sample in list:
            if sample[0] in sample_bitmap:
                sample_bitmap[sample[0]].append(sample)
            else:
                sample_bitmap[sample[0]] = [sample[1]]
        # POLICY ver1: generate training sequence if no loss of data for one day
        for date in sorted(sample_bitmap.keys()):
            if len(sample_bitmap[date]) >= 391:
                vecs = [smpl[2] for smpl in sorted(sample_bitmap[date], key=lambda t:t[1])]
                tr_vecs = [[vecs[0]] + vecs[i:i+time_interval-1] for i in range(1, 391-time_interval+2)]
                training_samples.append(tr_vecs)


        # POLICY ver2: generate training sequence if there are at least 390 samples in a day 
        # for date in sorted(sample_bitmap.keys()):
        #     if len(sample_bitmap[date]) >= 390:

        return training_samples
        
        pass

    # def get_stock_and_compliment(self, stock_symbol, time_interval):
    #     compliment_list = [e for e in self.stock_table_names if e != stock_symbol]
    #     return (self.get_serial_stock_sample([stock_symbol], time_interval),
    #             self.get_serial_stock_sample(compliment_list, time_interval))

    """ first stage helper methods """

    @staticmethod
    def _build_lookup_table(table_path):
        lookup_table = {}
        with open(table_path, "rb") as f:
            for line in f:
                line = line[0:-1]
                key = line.split(":")[0]
                values = line.split(":")[1]
                values = values.split(",")
                lookup_table[key] = values
        return lookup_table
    
    @staticmethod
    def _read_general_words(general_words_path):
        with open(general_words_path, 'r') as f:
            return f.read().strip().split(',')
    
    @staticmethod
    def _add_score (tweet_tups, key_words, score):
        '''
        add certain score to weight if some keywords are in the text
        :param tweet_tups: [(text, count, weight),...]
        '''

        for tup in tweet_tups:
            for key in key_words:
                if key.lower() in tup[0].lower():
                    tup[2] = tup[2] + score
                    break
        return tweet_tups


    def _get_stock_tweets(self, stock_symbol):
        """
        get all tweets relevant to the the given stock and generate weights of them
        :param stock_symbol: is the stock symbol
        :return: [(date, index, [(text, weight)])]
        """

        filtered_list = []
        for tuple in self.twitter_list:
            # tweet_tups = [(text, count, weight(0)),...]
            tweet_tups = [[t_tup[0],t_tup[1],0] for t_tup in tuple[2]]
            
            # filter against lookup_table
            tweet_tups = self._add_score(tweet_tups,self.lookup_table[stock_symbol],5 )
            tweet_tups = self._add_score(tweet_tups,self.general_words, 1)

            for t_tup in tweet_tups:
                t_tup[2] = t_tup[2] * math.log(t_tup[1])
            filtered_tweets = [(t_tup[0],t_tup[2]) for t_tup in tweet_tups if t_tup[2]>0]
            filtered_list.append((tuple[0], tuple[1], filtered_tweets))
        return filtered_list


    def _embedding(self, tweets_list):
        """
        turn [(date, index, [(text, weight)]] to [(date, index, vector)]
        :param tweets_list: [(date, index, [(text, weight)]]
        :return: [(date, index, vector)]
        """
        return map(lambda x: (x[0], x[1], self.model.event_embedding(x[2])), tweets_list)

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
        query = "SELECT Date,text,followers_count from Tweets WHERE followers_count > 10000" + partial_query
        query_result = self.twitter_db.query(query)

        # convert date format to (date, index), return [(date, index, text, count)]
        # date is datetime, index is int, text is string, count is int
        query_result = self._twitter_date_convert(query_result)

        # merge [(date, index, text, count)] to [(date, index, [(text, count)])]
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
                text_list.append((record[2], record[3]))
            key_tuple = (date, index, text_list)
            query_result.append(key_tuple)
        return query_result

    def _twitter_date_convert(self, query_result):
        """
        convert raw record's date to trading time date
        and make the indexing work easier
        :param query_result: [(date, text, count)]
        :return: data item is converted
        """

        # define a helper method
        def _time_convert(record):
            # get time of the record
            time = datetime.strptime(record[0], '%Y-%m-%d %H:%M:%S')
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
            new_record = (trading_date, trading_index, record[1], record[2])

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
        '../../resources/general_words.txt',
        None,
        '2018-04-23',
        0,
        '2018-04-25',
        391,
    )
    sample.get_serial_text_sample('aal',30)

