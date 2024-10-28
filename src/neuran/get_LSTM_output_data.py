import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from .LSTM_operations import predict_model
from ..neuran.input_data_for_training import get_days_data
from .db.products import get_list_of_products, get_list_of_products_by_category
from .db.sales import get_list_of_sales, get_sorted_sale_data, get_first_and_last_sale_dates
from .db.days import get_list_of_days
from .db.model_weights import get_scaled_input_data, get_scaler
from .residual_operations import predict_residuals_model
from .manipulate_input_data import get_input_data
from ..mongo.fake_day_data_creator import create_days_array
from .scaler import get_output_scaler
from .const import time_steps

def get_LSTM_output_data(user_name, requestData):
    products_data = []
    if requestData['subject']['type'] == 'byProduct':
        products_data = get_list_of_products(user_name, [requestData['subject']['product']])
    elif requestData['subject']['type'] == 'byCategory':
        products_data = get_list_of_products_by_category(user_name, [requestData['subject']['category']])
    elif requestData['subject']['type'] == 'allProducts':
        get_list_of_products(user_name)
    else:
        return []
    
    requested_start_date = datetime.strptime(requestData['relevantTime']['start'], '%Y-%m-%dT%H:%M:%S')
    requested_end_date = datetime.strptime(requestData['relevantTime']['end'], '%Y-%m-%dT%H:%M:%S')
    # requested_start_date -= timedelta(days=time_steps)
    
    # days_array = []
    # start = requested_start_date
    # end = requested_end_date
    # first_exist_date, last_exist_date = get_first_and_last_sale_dates(user_name)
    # if start < first_exist_date:
    #     if end <= first_exist_date:
    #         print('1')
    #         days_array.extend(create_days_array(start, end))
    #     else:
    #         print('2')
    #         end = first_exist_date
    #         days_array.extend(create_days_array(start, end))
    #         start = first_exist_date
    #         end = requested_end_date
    # if first_exist_date <= start < last_exist_date:
    #     if end <= last_exist_date:
    #         print('3')
    #         days_array.extend(get_list_of_days(user_name, start, end))
    #     else:
    #         print('4')
    #         end = last_exist_date
    #         days_array.extend(get_list_of_days(user_name, start, end))
    #         start = last_exist_date
    #         end = requested_end_date
    # if start >= last_exist_date:
    #     print('5')
    #     days_array.extend(create_days_array(start, end))
        
    # print('---------nurmal---------', create_days_array(start, end))
    # print('---------last---------', requested_end_date)
    # print('---------adjust---------', days_array)
    
    # days_array = get_list_of_days(user_name, requested_start_date, requested_end_date)
    list_of_days = get_list_of_days(user_name, requested_start_date, requested_end_date)
    scaler_name = 'scaler_nyc'
    X, Y = get_days_data(user_name, scaler_name, list_of_days)

    print(X.shape)
    print(Y.shape)
    
    predict_name = f'_E_nyc'
    predictions = predict_model(user_name, predict_name, X)
    
    Y_scaler_name = f'Y_{scaler_name}'
    Y_scaler = get_scaler(user_name, Y_scaler_name)
    prediction_data = Y_scaler.inverse_transform(predictions).reshape(-1,)
    
    residuals_predictions = predict_residuals_model(user_name, X)
    
    residuals_predictions = residuals_predictions.reshape(-1,).astype(int).tolist()
    prediction_data = prediction_data.astype(int).tolist()
    for i, day in enumerate(list_of_days):
        day['prediction'] = prediction_data[i]
        day['residual'] = residuals_predictions[i]
        day['minResidual'] = day['prediction'] - day['residual']
        day['maxResidual'] = day['prediction'] + day['residual']
        day['gap'] = abs(day['prediction'] - day['total_sales'])
        day['percentageGap'] = day['gap'] * 100 / day['total_sales']
        day['vacation'] = day['events']['vacation']
        day['holiday'] = day['events']['holiday']
        day['unusual'] = day['events']['unusual']
        del day['hourly_sales']
        
    return list_of_days
    
    # input_data = get_input_data(days_array, products_data)
    # n_products, n_days, n_features = input_data.shape
    
    # adjust_to_normalize_shape = input_data.reshape(n_products * n_days, n_features)
    # normalize_input_data = get_scaled_input_data(user_name, adjust_to_normalize_shape)
    
    # adjust_to_predict_shape = normalize_input_data.reshape(n_products, n_days, n_features)
    # adjust_to_predict_input_data = get_adjust_to_predict_input_data(adjust_to_predict_shape)
    
    # output_data = predict_model(user_name, adjust_to_predict_input_data)
    
    # output_scaler = get_output_scaler()
    # output_data = output_scaler.inverse_transform(output_data)
    
    
    # # TODO: return for route
    # sum_output_data = []
    # for date_data in output_data:
    #     sum_output_data.append(sum(date_data))
    
    # return sum_output_data

# from datetime import datetime, timedelta
# start_date = datetime.today()
# end_date = start_date + timedelta(weeks=4)
# fake_request_data = { 'products': ['Espresso'], 'relevantTime': { 'start': start_date, 'end': end_date } }
# print(get_LSTM_output_data('lizCafeteria', fake_request_data))

def get_adjust_to_predict_input_data(input_data):
    adjust_to_training = []
    
    for product in range(input_data.shape[0]):
        input_data_reshaped = []
        for i in range(time_steps, len(input_data[product])):
            input_data_reshaped.append(input_data[product, i - time_steps:i])
            
        adjust_to_training.extend(np.array(input_data_reshaped))

    return np.array(adjust_to_training)

