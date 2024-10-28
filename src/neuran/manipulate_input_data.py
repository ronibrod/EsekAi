import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from .const import time_steps

def get_input_data(list_of_days, list_of_products):
    input_data = []
    #TODO: change categories and products boolshit
    for product in list_of_products:
        category_index = product['categoryNum']
        product_index = product['productNum']
        input_data_per_product = get_input_data_per_day(list_of_days, product_index, category_index)

        input_data.append(input_data_per_product)
    return np.array(input_data)

def get_input_data_per_day(list_of_days, product_index, category_index):
    input_data_per_day = []
    
    for day in list_of_days:
        if day is not None:
            day_data = [
                day['date'].year,
                day['date'].month,
                day['date'].day,
                ((day['date'].weekday() + 1) % 7) + 1,
                day['min_temperature'],
                day['max_temperature'],
                day['rain'],
                day['events']['vacation'],
                day['events']['holiday'],
                day['events']['unusual'],
                product_index,
                category_index,
            ]
            input_data_per_day.append(day_data)
    return input_data_per_day
