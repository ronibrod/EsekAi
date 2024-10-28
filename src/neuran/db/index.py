from pymongo import MongoClient

# MongoDB configuration
client = MongoClient('localhost', 27017)

def get_model_weights_collection(user_name):
  return client[user_name]['model_weights']

def get_sales_collection(user_name):
  return client[user_name]['sale']

def get_days_collection(user_name):
  return client[user_name]['day']

def get_products_collection(user_name):
  return client[user_name]['product']

def get_users_collection():
    return client['users']['companies']
  