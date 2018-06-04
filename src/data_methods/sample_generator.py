from __future__ import print_function
from datetime import datetime
from twitter_methods import TwitterDatabase


class SampleGenerator:

    def __init__(self, twitter_db_path):
        self.date_index = 0
        self.text_index = 1
        self.seconds_in_minute = 60
        self.minutes_in_day = 391
        self.market_open_time = datetime.strptime('9:30AM', '%I:%M%p').time()
        self.market_close_time = datetime.strptime('4:00PM', '%I:%M%p').time()
        self.twitter_db = TwitterDatabase(twitter_db_path)

    def generate_tweets_list(self, query):

        # get query result from database
        query_result = self.twitter_db.query(query)

        # convert date of the result
        query_result = self._date_convert(query_result)

        # get start time and end time from all query result
        start_time, end_time = self._get_start_end_time(query_result)

        # convert index of the result
        query_result = self._index_convert(start_time, query_result)

        # return list of list of string
        return self._generate_text_bucket(start_time, end_time, query_result)

    def generate_stock_list(self, start_time, end_time):
        # return list of list of number
        pass

    """ helper methods below """

    def _date_convert(self, query_result):
        """
        convert raw record's date
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

            # convert time, deleting second
            if time.time() < self.market_open_time:
                time = time.replace(hour=open_hour, minute=open_min)
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




