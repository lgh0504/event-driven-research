from data_methods import SampleGenerator

""" test methods below """


def generate_tweets_list_test(database_path):
    query = "SELECT Date,text from Tweets WHERE followers_count > 100000 \
               AND DATETIME(Date) >=  '2018-04-23 09:10:00' AND DATETIME(Date) <= '2018-04-23 09:40:00'"
    text_bucket = SampleGenerator(database_path).generate_tweets_list(query)
    print(map(lambda x: len(x), text_bucket))


if __name__ == "__main__":
    from os import path

    current_path = path.dirname(path.abspath(__file__))
    root_path = path.dirname(path.dirname(current_path))
    db_path = path.join(root_path, "resources/twitter_database.db")
    generate_tweets_list_test(db_path)
