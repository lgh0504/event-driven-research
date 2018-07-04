from data_methods import SampleGenerator

""" test methods below """


def creator_test(twitter_database_path, stock_database_path):
    start_date = '2018-04-23'
    start_index = 0
    end_date = '2018-04-27'
    end_index = 390
    sample_generator = SampleGenerator(twitter_db_path, stock_db_path,
                                       start_date, start_index, end_date, end_index)
    twittwe_sample = sample_generator.get_serial_text_sample(1954)
    stock_sample = sample_generator.get_serial_stock_sample(['aal', 'aapl'], 1954)
    a = 1


if __name__ == "__main__":
    from os import path

    # parameter set up
    current_path = path.dirname(path.abspath(__file__))
    root_path = path.dirname(path.dirname(current_path))
    twitter_db_path = path.join(root_path, "resources/twitter_database.db")
    stock_db_path = path.join(root_path, "resources/nasdaq100_database.db")

    # test
    creator_test(twitter_db_path, stock_db_path)
