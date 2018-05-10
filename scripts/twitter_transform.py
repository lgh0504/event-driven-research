from __future__ import print_function
from datetime import datetime


# query result should be a list of (data, text)
def time_convert(query_result, start_time, end_time):

    # define a helper method
    def _time_convert(record):
        # get time
        time = datetime.strptime(record[0], '%Y-%m-%d %H:%M:%S')

        # convert time
        if time.time() < start_time:
            time = time.replace(hour=9, minute=30)
        if time.time() > end_time:
            time = time.replace(hour=16, minute=0)
        time = time.replace(second=0)

        # generate new record
        y = [time, record[1]]

        return y

    # using the helper method
    return map(_time_convert, query_result)


# given a formatted query result
# return buckets of text
def generate_data_bucket(query_result, bucket_number, start_time):
    seconds_in_minute = 60
    minutes_in_day = 391

    def date_to_key(record):
        time = record[0]
        delta_date = time.date() - start_time.date()
        delta_day = delta_date.days
        delta_min = delta_date.seconds / seconds_in_minute
        time_index = delta_day * minutes_in_day + delta_min
        y = [time_index, record[1]]
        return y

    query_result = map(date_to_key, query_result)
    data_buckets = [[]] * bucket_number

    for item in query_result:
        index = item[0]
        text = item[1]
        data_buckets[index].append(text)

    return data_buckets




