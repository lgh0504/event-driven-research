import os
from os import path

# get root path
project_root = path.dirname(path.abspath(__file__))
resources_root = path.join(project_root, "resources")
src_root = path.join(project_root, "src")
database_root = path.join(resources_root, "database")
lookup_root = path.join(resources_root, "lookup_table")

# set file specific path
twitter_database = path.join(database_root, "big_twitter_database.db")
stock_database = path.join(database_root, "new_nasdaq100_database.db")
track_words = path.join(lookup_root, "track_words.txt")
filter_table = path.join(lookup_root, "filter_table.txt")
word_vec_model = path.join(lookup_root, "GoogleNews-vectors-negative300.bin")

# environment variable dictionary
environment_variables = {
    "PROJECT_ROOT": project_root,
    "RESOURCES_ROOT": resources_root,
    "PYTHONPATH": src_root,
    "DATABASE_ROOT": database_root,
    "LOOKUP_ROOT": lookup_root
}

# set up the environment variable
for key, value in environment_variables.items():
    print('export '+key+'='+value+":$"+key)
