from __future__ import print_function

import pickle
from datetime import datetime

from data_methods.preprocess_methods import generate_data_bucket
from data_methods.preprocess_methods import time_convert
from data_methods.preprocess_methods.text2Vector import *
from data_methods.twitter_data_methods import TwitterDatabase

# set up path
current_path = path.dirname(path.abspath(__file__))
parent_path = path.dirname(current_path)
db_path = path.join(parent_path, 'resources/twitter_database.db')

# parameters
minutes_in_day = 391
seconds_in_minute = 60
start_time = datetime.strptime('9:30AM', '%I:%M%p').time()
end_time = datetime.strptime('4:00PM', '%I:%M%p').time()
query = "SELECT Date,text from Tweets WHERE followers_count > 100000 " \
        "AND DATETIME(Date) >= '2018-04-23 09:00:00' AND DATETIME(Date) <= '2018-04-27 16:00:01'"

# get data from database
db = TwitterDatabase(db_path=db_path)

# querying from data base
print("executing the query...")
query_result = db.query(query)
print("query done!")

# format the time
query_result = time_convert(query_result, start_time, end_time)
time_series = map(lambda record: record[0], query_result)

# generate data buckets (one bucket for one minute, containing all texts in this minute)
max_time = max(time_series)
min_time = min(time_series)
time_duration = max_time - min_time
bucket_num = time_duration.days * minutes_in_day + time_duration.seconds / seconds_in_minute + 1
bucket_num = max(bucket_num, 1)

# generate  data bucket
text_buckets = generate_data_bucket(query_result, bucket_num, min_time)

# merge all texts in a bucket into a single event vector
# word to vector first
print("map working...")
text_buckets = map(texts2vectors, text_buckets)
print("map done...")

# merge all text vector
event_vectors = []
for texts in text_buckets:
    text_vectors = map(vectors_mean, texts)
    event_vector = vectors_mix(text_vectors)
    event_vectors.append(event_vector)
pickle.dump(event_vectors, open(path.join(parent_path, "resources/training_data/event_vectors.p"), "wb"))
