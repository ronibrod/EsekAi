import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from .db.days import get_list_of_days
from .db.products import get_list_of_products
from .db.sales import get_first_and_last_sale_dates
from .manipulate_input_data import get_input_data
from .scaler import get_output_scaler, scale_down
from .db.model_weights import get_scaled_input_data
from .const import time_steps, user_name

def get_days_data(user_name, scaler_name, list_of_days):
    
    # hourly_data = []
    # for day in list_of_days:
    #     for hour_index, hour in enumerate(day['hourly_sales']):
    #         data = {
    #             '_id': day['_id'],
    #             'date': day['date'],
    #             'day_of_week': day['day_of_week'],
    #             'min_temperature': day['min_temperature'],
    #             'max_temperature': day['max_temperature'],
    #             'mean_temperature': day['mean_temperature'],
    #             'feel_min_temp': day['feel_min_temp'],
    #             'feel_max_temp': day['feel_max_temp'],
    #             'feel_mean_temp': day['feel_mean_temp'],
    #             'rain': day['rain'],
    #             'snow': day['snow'],
    #             'windspeed': day['windspeed'],
    #             'cloudcover': day['cloudcover'],
    #             'humidity': day['humidity'],
    #             'severerisk': day['severerisk'],
    #             'events': day['events'],
    #             'range_time': day['range_time'],
    #             'hour': hour_index,
    #             'total_sales': hour['total']
    #         }
    #         hourly_data.append(data)
    
    # Convert hourly data to DataFrame
    # X = pd.DataFrame(hourly_data)
    X = pd.DataFrame(list_of_days)
    
    # Drop unnecessary columns
    X.drop(['_id'], axis=1, inplace=True)
    
    # Convert 'date' to datetime and extract date components
    X['date'] = pd.to_datetime(X['date'])
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    
    # Extract event-related columns
    X['vacation'] = X['events'].apply(lambda x: x['vacation'])
    X['holiday'] = X['events'].apply(lambda x: x['holiday'])
    X['unusual'] = X['events'].apply(lambda x: x['unusual'])
    
    # Drop 'events' column as it's no longer needed
    X.drop(['events'], axis=1, inplace=True)
    
    # Define desired columns
    desired_columns = [
        'year',
        'month',
        'day',
        'day_of_week',
        'min_temperature',
        'max_temperature',
        'mean_temperature',
        'feel_min_temp',
        'feel_max_temp',
        'feel_mean_temp',
        'rain',
        'snow',
        'windspeed',
        'cloudcover',
        'humidity',
        'severerisk',
        'vacation',
        'holiday',
        'unusual',
        'range_time',
        # 'hour',
        'total_sales'
    ]
    
    # Select desired columns
    X = X[desired_columns]
    
    # Separate features (X) and target variable (Y)
    Y = X[['total_sales']]
    X.drop(['total_sales'], axis=1, inplace=True)
    
    # Backfill missing values
    # X.bfill(inplace=True)
    
    # print(Y.head(24))
    
    # Scale the features and target variable
    X_scaler_name = f'X_{scaler_name}'
    Y_scaler_name = f'Y_{scaler_name}'
    X_scaled = get_scaled_input_data(user_name, X_scaler_name, X)
    Y_scaled = get_scaled_input_data(user_name, Y_scaler_name, Y)
    # X_scaler = MinMaxScaler(feature_range=(0, 1))
    # X_scaled = X_scaler.fit_transform(X)
    # Y_scaler = MinMaxScaler(feature_range=(0, 1))
    # Y_scaled = Y_scaler.fit_transform(Y)
    
    return X_scaled, Y_scaled
    
    
    
# get_days_data(user_name)
    # input_data_per_day = []
    
    # for day in list_of_days:
    #     if day is not None:
    #         day_data = {
    #             'year': day['date'].year,
    #             day['date'].month,
    #             day['date'].day,
    #             ((day['date'].weekday() + 1) % 7) + 1,
    #             day['min_temperature'],
    #             day['max_temperature'],
    #             day['rain'],
    #             day['events']['vacation'],
    #             day['events']['holiday'],
    #             day['events']['unusual'],
    #             product_index,
    #             category_index,
    #         }
    #         input_data_per_day.append(day_data)

def create_input_data_for_training(user_name):
    first_date, last_date = get_first_and_last_sale_dates(user_name)
    list_of_days = get_list_of_days(user_name, first_date, last_date)
    list_of_products = get_list_of_products(user_name)
    
    input_data = get_input_data(list_of_days, list_of_products)
    n_products, n_days, n_features = input_data.shape
    
    adjust_to_normalize_shape = input_data.reshape(n_products * n_days, n_features)
    normalize_input_data = get_scaled_input_data(user_name, adjust_to_normalize_shape)
    
    adjust_to_training_shape = normalize_input_data.reshape(n_products, n_days, n_features)
    adjust_to_training_input_data = get_adjust_to_training_input_data(adjust_to_training_shape)
    
    return adjust_to_training_input_data

def get_adjust_to_training_input_data(input_data):
    adjust_to_training = []
    
    for product in range(input_data.shape[0]):
        input_data_reshaped = []
        for i in range(time_steps, len(input_data[product])):
            input_data_reshaped.append(input_data[product, i - time_steps:i])
            
        adjust_to_training.extend(np.array(input_data_reshaped))

    return np.array(adjust_to_training)
