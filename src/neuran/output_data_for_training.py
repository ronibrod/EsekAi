import numpy as np
from .db.sales import get_list_of_sales, get_sorted_sale_data, get_first_and_last_sale_dates
from .scaler import get_output_scaler
from .db.products import get_list_of_products
from .const import time_steps, hour_steps

def get_output_data(user_name):
    list_of_products = get_list_of_products(user_name)
    first_date, last_date = get_first_and_last_sale_dates(user_name)
    
    output_data = []
    for product in list_of_products:
        list_of_sales = get_list_of_sales(user_name, product)
        sales_counts_per_hour = get_sorted_sale_data(list_of_sales, first_date, last_date)
        sales_sum_by_time_division = [len(hour) for hour in sales_counts_per_hour]
        
        output_scaler = get_output_scaler()
        normalize_output_data = output_scaler.transform(sales_sum_by_time_division)
        reshape_output_data = get_reshape_output_data(normalize_output_data)
        output_data.extend(reshape_output_data)
        
    return np.array(output_data)

def get_reshape_output_data(output_data):
    output_data = output_data[:(len(output_data) // (hour_steps)) * (hour_steps)]
    output_data = np.array(output_data).reshape(-1, hour_steps)
    output_data = output_data[time_steps:]
    
    return np.array(output_data)
    # output_data_reshaped = []
    # for i in range(0, len(output_data) - time_steps + 1, time_steps):
    #     output_data_reshaped.append(output_data[i:i + time_steps])

    # return np.array(output_data_reshaped)
