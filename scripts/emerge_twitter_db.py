import sqlite3
from os import path
import sys

current_path = path.dirname(path.abspath(__file__))
root_path = path.dirname(current_path)
sys.path.append(root_path)

db_old_path = path.join(root_path, r'resources/twitter_database.db')
db_new_path = path.join(root_path, r'resources/tweets.db')

new_conn = sqlite3.connect(db_new_path)
c = new_conn.cursor()
#print("Before insert: new",c.execute("SELECT COUNT(*) FROM main.Tweets").fetchall())
c.execute("ATTACH DATABASE ? AS old_db", [db_old_path])
#print("Old DB: ",c.execute("SELECT COUNT(*) FROM old_db.Tweets").fetchall())
c.execute("INSERT INTO main.Tweets SELECT * FROM old_db.Tweets")
#print("After insert: ",c.execute("SELECT COUNT(*) FROM main.Tweets").fetchall())
new_conn.commit()
new_conn.close()