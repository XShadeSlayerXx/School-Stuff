import pymongo
from pprint import pprint

MONGO_URL = ''

client = pymongo.MongoClient(MONGO_URL)
db = client.admin

status = db.command("serverStatus")
pprint(status)