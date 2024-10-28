from pymongo import ASCENDING
from .index import get_products_collection

def get_list_of_products(user_name, products=[]):
    products_collection = get_products_collection(user_name)
    
    query = {}
    if len(products):
        query['name'] = { '$in': products }
    
    return list(products_collection.find(query).sort([('category', ASCENDING), ('name', ASCENDING)]))

def get_list_of_products_by_category(user_name, categories=[]):
    products_collection = get_products_collection(user_name)

    query = {}
    if len(categories):
        query['category'] = { '$in': categories }
        
    return list(products_collection.find(query).sort([('category', ASCENDING), ('name', ASCENDING)]))
