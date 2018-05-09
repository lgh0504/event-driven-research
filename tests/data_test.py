from __future__ import print_function
from os import path
import pickle
current_path = path.dirname(path.abspath(__file__))
parent_path = path.dirname(current_path)
resources_path = path.join(parent_path, 'resources/training_data/stock_prices.p')

stock_infos = pickle.load(open(resources_path))

print(len(stock_infos[0]) / 390)