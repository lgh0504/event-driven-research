from __future__ import print_function
import sqlite3
import time
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import json


class KeyWordFilter:
    """
    Filter tweets from twitter streaming API by keywords
    """

    def __init__(self, key_words):
        """
        create a filter
        :param key_words: is the key words list used to filter the tweets
        """
        self._key_words = [word.lower() for word in key_words]

    def filter(self, tweet):
        """
        judge if a tweet should be filtered
        :param tweet: a dictionary format of a tweet
        :return: boolean true if tweet should be kept, else false
        """
        try:
            text = tweet['text']
        except KeyError:
            return False
        text = text.lower()
        for word in self._key_words:
            if word in text:
                return True
        return False


class TwitterFormat:
    """
    Format the tweet, pick useful information
    """

    def __init__(self):
        """
        do nothing
        """
        pass

    @staticmethod
    def format(twitter):
        """
        format the tweet
        :param twitter: a dictionary format of a tweet
        :return: a formatted dictionary
        """
        formatted_twitter = {}
        try:
            formatted_twitter['Date'] = time.strftime('%Y-%m-%d %H:%M:%S',
                                                      time.strptime(twitter['created_at'],
                                                                    '%a %b %d %H:%M:%S +0000 %Y'))
            formatted_twitter['text'] = twitter['text']
            formatted_twitter['user_id'] = twitter['user']['id']
            formatted_twitter['followers_count'] = twitter['user']['followers_count']
            formatted_twitter['listed_count'] = twitter['user']['listed_count']
            formatted_twitter['statuses_count'] = twitter['user']['statuses_count']
            formatted_twitter['friends_count'] = twitter['user']['friends_count']
            formatted_twitter['favourites_count'] = twitter['user']['favourites_count']
        except:
            raise KeyError
        return formatted_twitter


class TwitterDatabase:
    """
    Methods related to tweet database
    """

    def __init__(
            self,
            db_path,
            create_table_sql='''CREATE TABLE IF NOT EXISTS Tweets (
            Date DATETIME,
            text TEXT,
            followers_count INT,
            listed_count INT,
            statuses_count INT,
            friends_count INT,
            favourites_count INT
            )'''):
        """
        create the link to the database and create table if not exists
        :param db_path: path to database
        :param create_table_sql: SQL to create the default table
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table(create_table_sql)
        self.insert_count = 0

    def __del__(self):
        self.conn.commit()
        self.conn.close()
        print("Tweets Database closed")

    def _create_table(self, create_table_sql):
        self.cursor.execute(create_table_sql)

    def insert_dict(
            self,
            tweet,
            table_name='Tweets',
            keys={'Date', 'text', 'followers_count', 'listed_count', 'statuses_count',
                  'friends_count', 'favourites_count'},
            batch_mode=True
    ):
        """
        insert a dictionary to the database
        :param tweet: dictionary of a tweet
        :param table_name: is the table name to insert
        :param keys: are the keys of the dictionary
        :param batch_mode: performance choice
        :return: nothing
        """
        row = [tweet[key] for key in keys]
        row_format = '(' + ','.join(['?'] * len(row)) + ')'
        self.cursor.execute('INSERT INTO %s VALUES %s' % (table_name, row_format), row)

        if not batch_mode:
            self.conn.commit()
            return

        self.insert_count += 1
        if self.insert_count % 10 == 0:
            self.conn.commit()
            self.insert_count = 0
            print("Commit 10000 inserts...")

    def query(self, query):
        """
        execute a query
        :param query: a SQL format string
        :return: a list of query result, each result is a tuple
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()


class _MyStreamListener(StreamListener):
    """
    a listener dealing with streaming data
    """

    def __init__(self, tweet_filter, tweet_formator, tweet_db):
        super(_MyStreamListener, self).__init__()
        self.tweet_filter = tweet_filter
        self.tweet_formator = tweet_formator
        self.tweet_db = tweet_db

    def on_data(self, data):
        """
        what action to perform when get a data
        :param data: the a string of tweet
        :return: True
        """
        try:
            data = json.loads(data)
            if self.tweet_filter.filter(data):
                data = self.tweet_formator.format(data)
                self.tweet_db.insert_dict(data)
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        """
        what action to perform when get an error
        :param status:
        :return:
        """
        print(status)
        return True


class TwitterStreaming:
    """
    Using this class to run the pipeline of twitter data processing
    including data get, filter, format and insert into database
    """

    def __init__(self, tokens, track, tweet_filter, tweet_formator, tweet_db):
        """
        set how to run the pipeline
        :param tokens: dictionary of all tokens
        :param track: keywords to track
        :param tweet_filter: an instance of filter class
        :param tweet_formator: an instance of format class
        :param tweet_db: an instance of database class
        """
        access_token = tokens['access_token']
        access_token_secret = tokens['access_token_secret']
        consumer_key = tokens['consumer_key']
        consumer_secret = tokens['consumer_secret']
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.track = track
        self.twitter_stream = Stream(auth, _MyStreamListener(tweet_filter=tweet_filter,
                                                             tweet_formator=tweet_formator, tweet_db=tweet_db))

    def run(self):
        """
        run the pipeline, this is unstopable
        :return: nothing
        """
        while True:
            try:
                self.twitter_stream.filter(track=self.track, languages=["en"])
            except:
                continue
