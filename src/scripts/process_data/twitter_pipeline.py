import os
from os import path
import data_methods as dm

if __name__ == '__main__':

    # set path
    lookup_root = os.environ['LOOKUP_ROOT']
    database_root = os.environ['DATABASE_ROOT']
    track_path = path.join(lookup_root, 'TRACK_WORDS.txt')
    database_path = path.join(database_root, 'twitter_database.db')

    # streaming config
    tokens = {
        'access_token': "982333914201993216-3lUuCAaUCYgkB4kpkRz1tzL24veeYeX",
        'access_token_secret': "3XSDsJXSmUXwOwOlEflXYFTCXxIDOdhMQ4kNNzuTIRXkB",
        'consumer_key': "bz58HpjCEXS0kgn21Rj3qcvNo",
        'consumer_secret': "LjcezoypAs4Rjgmsd32bd8dB6tkGg7c6UvpIQ66hUi99EJYyPB"
    }

    with open(track_path, 'r') as f:
        stock_symbol = f.read()
        track = stock_symbol.split(",")

    # filter config
    tweet_filter = dm.KeyWordFilter(track)

    # format config
    tweet_formator = dm.TwitterFormat()

    # database config
    tweet_database = dm.TwitterDatabase(database_path)

    # pipeline config
    twitter_data_pipeline = dm.TwitterStreaming(tokens, track, tweet_filter, tweet_formator, tweet_database)

    # run the pipeline
    twitter_data_pipeline.run()
