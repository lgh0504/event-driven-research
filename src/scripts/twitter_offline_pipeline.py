from os import path
import json
import data_methods as dm

if __name__ == '__main__':
    current_path = path.dirname(path.abspath(__file__))
    root_path = path.dirname(current_path)

    twitter_path = path.join(root_path, r"resources/NASDQ100_tweets.json")
    track_path = path.join(root_path, r'resources/TRACK_WORDS.txt')
    database_path = path.join(root_path, r'resources/twitter_database.db')

    fail_count = 0
    fail_idx = []
    line_cnt = 0

    with open(track_path, 'r') as f:
        stock_symbol = f.read()
        track = stock_symbol.split(",")

    tweet_filter = dm.KeyWordFilter(track)
    tweet_formator = dm.TwitterFormat()
    tweet_database = dm.TwitterDatabase(database_path)

    with open(twitter_path, 'r') as f:
        for line in f:
            line_cnt += 1
            # try:
            data = json.loads(line)
            if tweet_filter.filter(data):
                data = tweet_formator.format(data)
                tweet_database.insert_dict(data)
            # except BaseException as e:
            #     fail_count += 1
            #     fail_idx.append(line_cnt)
            #     print("Error: %s" % str(e))

    print(str(fail_count) + " tweets failed in pipeline")
