import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print('--- Getting Ready ---')

import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from .LSTM import model
from .residuals_model import residuals_model
from .LSTM_operations import train_model, predict_model
from .db.model_weights import insert_model_data, save_new_model_data, get_scaled_input_data, get_scaler
from .input_data_for_training import get_days_data, create_input_data_for_training
from .db.days import get_list_of_days
from .db.sales import get_first_and_last_sale_dates
from .output_data_for_training import get_output_data
from .const import user_name, user_name9, time_steps


def run(user_name):
    first_date, last_date = get_first_and_last_sale_dates(user_name)
    list_of_days = get_list_of_days(user_name, first_date, last_date)
    scaler_name = 'scaler_nyc'
    X, Y = get_days_data(user_name, scaler_name, list_of_days)

    print(X.shape)
    print(Y.shape)

    # -------------LSTM----------------
    # X_lstm = []
    # Y_lstm = []
    # for i in range(len(X) - time_steps):
    # 	sequence = X[i:i + time_steps]
    # 	label = Y[i + time_steps]
    # 	X_lstm.append(sequence)
    # 	Y_lstm.append(label)
    # X = np.array(X_lstm)
    # Y = np.array(Y_lstm)
    # -----------------------------------

    # split_index = int(0.8 * len(X))
    # split_index = 140
    # X_train = X[:split_index]
    # X_test = X[split_index:]
    # Y_train = Y[:split_index]
    # Y_test = Y[split_index:]
    
    X_train = X
    X_test = X
    Y_train = Y
    Y_test = Y

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



    # X_train = np.array(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]))
    # X_test = np.array(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))

    # X_train = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]])
    # Y_train = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    # X_train = X[:320]
    # Y_train = Y[:320]

    # print(len(X_train)) # 1290
    # train_model(user_name, X_training, Y_training)
    epochs_size = int(len(X_train) * 8)
    batch_size = math.ceil(math.sqrt(len(X_train)))

    predict_name = f'_E_nyc'
    insert_model_data(user_name, predict_name)
    start_time = time.time()
    # model.fit(X_train, Y_train, epochs=epochs_size*2, batch_size=3) # verbose=0
    end_time = time.time()
    save_new_model_data(user_name, predict_name)
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    

    elapsed_time = end_time - start_time
    print(f"\nElapsed time: {elapsed_time} seconds")


    # print_errors(X_train, X_test, Y_train, Y_test, Y_scaler, train_predictions, test_predictions)




    Y_scaler_name = f'Y_{scaler_name}'
    Y_scaler = get_scaler(user_name, Y_scaler_name)
    inverse_Y_train = Y_scaler.inverse_transform(Y_train)
    inverse_train_predictions = Y_scaler.inverse_transform(train_predictions)
    inverse_Y_test = Y_scaler.inverse_transform(Y_test)
    inverse_test_predictions = Y_scaler.inverse_transform(test_predictions)
    
    






    # fig = plt.figure(figsize=(15, 16))
    # fig1_1 = fig.add_subplot(4, 2, 1)
    # fig1_2 = fig.add_subplot(4, 2, 2)
    # fig2_1 = fig.add_subplot(4, 2, 3)
    # fig2_2 = fig.add_subplot(4, 2, 4)
    # fig3_1 = fig.add_subplot(4, 2, 5)
    # fig3_2 = fig.add_subplot(4, 2, 6)
    # fig4_1 = fig.add_subplot(4, 2, 7)
    # fig4_2 = fig.add_subplot(4, 2, 8)

    # fig1_1.scatter(inverse_Y_train, inverse_train_predictions)
    # fig1_1.plot([min(inverse_Y_train), max(inverse_Y_train)], [min(inverse_Y_train), max(inverse_Y_train)], color='red')
    # fig1_1.set_xlabel('True Values')
    # fig1_1.set_ylabel('Predictions')
    # fig1_1.set_title('Train: Predictions vs True Values')

    # fig1_2.scatter(inverse_Y_test, inverse_test_predictions)
    # fig1_2.plot([min(inverse_Y_test), max(inverse_Y_test)], [min(inverse_Y_test), max(inverse_Y_test)], color='red')
    # fig1_2.set_xlabel('True Values')
    # fig1_2.set_ylabel('Predictions')
    # fig1_2.set_title('Test: Predictions vs True Values')

    # train_residuals = inverse_Y_train - inverse_train_predictions
    # fig2_1.scatter(inverse_train_predictions, train_residuals)
    # fig2_1.hlines(y=0, xmin=min(inverse_train_predictions), xmax=max(inverse_train_predictions), color='red')
    # fig2_1.set_xlabel('Predictions')
    # fig2_1.set_ylabel('Residuals')
    # fig2_1.set_title('Train: Residuals vs Predictions')

    # test_residuals = inverse_Y_test - inverse_test_predictions
    # fig2_2.scatter(inverse_test_predictions, test_residuals)
    # fig2_2.hlines(y=0, xmin=min(inverse_test_predictions), xmax=max(inverse_test_predictions), color='red')
    # fig2_2.set_xlabel('Predictions')
    # fig2_2.set_ylabel('Residuals')
    # fig2_2.set_title('Test: Residuals vs Predictions')

    # fig3_1.hist(train_residuals, bins=20)
    # fig3_1.set_xlabel('Error')
    # fig3_1.set_ylabel('Frequency')
    # fig3_1.set_title('Train: Distribution of Errors')

    # fig3_2.hist(test_residuals, bins=20)
    # fig3_2.set_xlabel('Error')
    # fig3_2.set_ylabel('Frequency')
    # fig3_2.set_title('Test: Distribution of Errors')

    # fig4_1.plot(inverse_Y_train[:100], label='True Values', color='blue')
    # fig4_1.plot(inverse_train_predictions[:100], label='Predictions', color='orange')
    # fig4_1.set_xlabel('Index')
    # fig4_1.set_ylabel('Values')
    # fig4_1.set_title('Train: True Values and Predictions Over Time')
    # fig4_1.legend()

    # fig4_2.plot(inverse_Y_test[:100], label='True Values', color='blue')
    # fig4_2.plot(inverse_test_predictions[:100], label='Predictions', color='orange')
    # fig4_2.set_xlabel('Index')
    # fig4_2.set_ylabel('Values')
    # fig4_2.set_title('Test: True Values and Predictions Over Time')
    # fig4_2.legend()

    # plt.tight_layout()
    # plt.show()










    residual_name = '_residual_nyc'
    scaler_name = 'residuals_scaler_nyc'
    predict = model.predict(X)
    residuals = abs(Y_scaler.inverse_transform(Y) - Y_scaler.inverse_transform(predict))
    residuals_scaled = get_scaled_input_data(user_name, scaler_name, residuals)
    residuals_scaler = get_scaler(user_name, scaler_name)
    # residuals_scaler = MinMaxScaler(feature_range=(0, 1))
    # residuals_scaled = residuals_scaler.fit_transform(residuals)
    
    # X_residuals_train, X_residuals_test, Y_residuals_train, Y_residuals_test = train_test_split(X, residuals_scaled, test_size=0.2, random_state=42)
    X_residuals_train = X
    X_residuals_test = X
    Y_residuals_train = residuals_scaled
    Y_residuals_test = residuals_scaled
    
    insert_model_data(user_name, residual_name)
    residuals_model.fit(X_residuals_train, Y_residuals_train, epochs=epochs_size, batch_size=3)
    save_new_model_data(user_name, residual_name)
    
    train_residuals_predictions = model.predict(X_residuals_train)
    test_residuals_predictions = model.predict(X_residuals_test)
    
    inverse_predict_residuals = residuals_scaler.inverse_transform(Y_residuals_train)
    inverse_train_residuals_predictions = residuals_scaler.inverse_transform(train_residuals_predictions)
    inverse_Y_residuals_test = residuals_scaler.inverse_transform(Y_residuals_test)
    inverse_test_residuals_predictions = residuals_scaler.inverse_transform(test_residuals_predictions)
    
    # print(inverse_train_residuals_predictions)
    # print(inverse_Y_residuals_train)













    # print('E', Y_scaler.inverse_transform(Y_train)[:12].flatten().astype(int))
    # print('P', Y_scaler.inverse_transform(train_predictions)[:12].flatten().astype(int))
    # print('RE', inverse_Y_residuals_train[:12].flatten().astype(int))
    # print('RP', inverse_train_residuals_predictions[:12].flatten().astype(int))

    # print('E', Y_scaler.inverse_transform(Y_test)[:12].flatten().astype(int))
    # print('P', Y_scaler.inverse_transform(test_predictions)[:12].flatten().astype(int))
    # print('RE', inverse_Y_residuals_test[:12].flatten().astype(int))
    # print('RP', inverse_test_residuals_predictions[:12].flatten().astype(int))
    
    print_errors(X_train, X_test, Y_train, Y_test, Y_scaler, train_predictions, test_predictions)
    
    min_train_predictions = inverse_train_predictions - (inverse_train_residuals_predictions / 2)
    max_train_predictions = inverse_train_predictions + (inverse_train_residuals_predictions / 2)
    min_test_predictions = inverse_test_predictions - inverse_test_residuals_predictions
    max_test_predictions = inverse_test_predictions + inverse_test_residuals_predictions
    
    train_bigger = inverse_train_residuals_predictions[(inverse_Y_train - max_train_predictions) > 0]
    train_smaller = inverse_train_residuals_predictions[(inverse_Y_train - min_train_predictions) < 0]
    br2 = sum(abs(train_bigger))
    sr2 = sum(abs(train_smaller))
    r2 = math.sqrt((br2 + sr2) / len(inverse_train_predictions))
    print(train_bigger)
    print(train_smaller)
    print('r2:', r2)
    print('Average: ', sum(inverse_train_residuals_predictions) / len(inverse_train_residuals_predictions))
    
    
    # print(np.ndim(min_train_predictions))
    # print(np.ndim(max_train_predictions))
    min_train_predictions = np.ravel(min_train_predictions)
    max_train_predictions = np.ravel(max_train_predictions)
    min_test_predictions = np.ravel(min_test_predictions)
    max_test_predictions = np.ravel(max_test_predictions)
    
    # print(min_test_predictions[:100])
    
    
    
    fig = plt.figure(figsize=(15, 5))
    fig1 = fig.add_subplot(1, 2, 1)
    fig2 = fig.add_subplot(1, 2, 2)
    
    fig1.plot(inverse_Y_train[:100], label='True Values', color='orange')
    fig1.plot(min_train_predictions[:100], label='Min Predictions', color='blue')
    fig1.plot(max_train_predictions[:100], label='Max Predictions', color='blue')
    fig1.fill_between(range(len(np.ravel(min_train_predictions[:100]))),
        min_train_predictions[:100],
        max_train_predictions[:100],
        color='blue', alpha=0.3)
    fig1.set_xlabel('Index')
    fig1.set_ylabel('Values')
    fig1.set_title('Train: True Values and Predictions Over Time')
    fig1.legend()
    
    fig2.plot(inverse_Y_test[:100], label='True Values', color='orange')
    fig2.plot(min_test_predictions[:100], label='Min Predictions', color='blue')
    fig2.plot(max_test_predictions[:100], label='Max Predictions', color='blue')
    fig2.fill_between(range(len(np.ravel(min_test_predictions[:100]))),
        min_test_predictions[:100],
        max_test_predictions[:100],
        color='blue', alpha=0.3)
    fig2.set_xlabel('Index')
    fig2.set_ylabel('Values')
    fig2.set_title('Test: True Values and Predictions Over Time')
    fig2.legend()

    plt.tight_layout()
    plt.show()







def print_errors(X_train, X_test, Y_train, Y_test, Y_scaler, train_predictions, test_predictions):
    tf_mse = MeanSquaredError()
    SUM_OF = 24

    print('\n---------------- train -----------------')
    print(len(train_predictions))
    train_tf_mse = tf_mse(Y_train, train_predictions)
    train_skl_mse = mean_squared_error(Y_train, train_predictions)
    train_skl_mae = mean_absolute_error(Y_train, train_predictions)
    train_skl_rmse = np.sqrt(train_skl_mse)
    train_sse = sum([(y - y_predic)**2 for y, y_predic in zip(Y_train, train_predictions)])
    train_r2 = r2_score(Y_train, train_predictions)

    print('tf_mse: ', train_tf_mse)
    print("skl_mse:", train_skl_mse)
    print("skl_mae:", train_skl_mae)
    print("skl_rmse:", train_skl_rmse)
    print("SSE div:", train_sse / len(Y_train))
    print("SSE all:", train_sse)
    print("R^2 Score:", train_r2)
    print('E', Y_scaler.inverse_transform(Y_train)[:12].flatten().astype(int), 'E', Y_scaler.inverse_transform(Y_train)[-12:].flatten().astype(int))
    print('P', Y_scaler.inverse_transform(train_predictions)[:12].flatten().astype(int), 'P', Y_scaler.inverse_transform(train_predictions)[-12:].flatten().astype(int))
    print(f'sum of day E:', np.sum(Y_scaler.inverse_transform(Y_train)[:SUM_OF].flatten().astype(int)),
          f' | sum of week E:', np.sum(Y_scaler.inverse_transform(Y_train)[:(7 * SUM_OF)].flatten().astype(int)),
          f' | sum of month E:', np.sum(Y_scaler.inverse_transform(Y_train)[:(7 * 30 * SUM_OF)].flatten().astype(int)),
          f' | sum of all E:', np.sum(Y_scaler.inverse_transform(Y_train).flatten().astype(int)))
    print(f'sum of day P:', np.sum(Y_scaler.inverse_transform(train_predictions)[:SUM_OF].flatten().astype(int)),
          f' | sum of week P:', np.sum(Y_scaler.inverse_transform(train_predictions)[:(SUM_OF * 7)].flatten().astype(int)),
          f' | sum of month P:', np.sum(Y_scaler.inverse_transform(train_predictions)[:(SUM_OF * 7 * 30)].flatten().astype(int)),
          f' | sum of all P:', np.sum(Y_scaler.inverse_transform(train_predictions).flatten().astype(int)))






    print('\n---------------- test -----------------')
    print(len(test_predictions))
    test_tf_mse = tf_mse(Y_test, test_predictions)
    test_skl_mse = mean_squared_error(Y_test, test_predictions)
    test_skl_mae = mean_absolute_error(Y_test, test_predictions)
    test_skl_rmse = np.sqrt(test_skl_mse)
    test_sse = sum([(y - y_predic)**2 for y, y_predic in zip(Y_test, test_predictions)])
    test_r2 = r2_score(Y_test, test_predictions)

    print('tf_mse: ', test_tf_mse)
    print("skl_mse:", test_skl_mse)
    print("skl_mae:", test_skl_mae)
    print("skl_rmse:", test_skl_rmse)
    print("SSE div:", test_sse / len(Y_test))
    print("SSE all:", test_sse)
    print("R^2 Score:", test_r2)
    print('E', Y_scaler.inverse_transform(Y_test)[:12].flatten().astype(int), 'E', Y_scaler.inverse_transform(Y_test)[-12:].flatten().astype(int))
    print('P', Y_scaler.inverse_transform(test_predictions)[:12].flatten().astype(int), 'P', Y_scaler.inverse_transform(test_predictions)[-12:].flatten().astype(int))
    print(f'sum of day E:', np.sum(Y_scaler.inverse_transform(Y_test)[:SUM_OF].flatten().astype(int)),
          f' | sum of week E:', np.sum(Y_scaler.inverse_transform(Y_test)[:(7 * SUM_OF)].flatten().astype(int)),
          f' | sum of month E:', np.sum(Y_scaler.inverse_transform(Y_test)[:(7 * 30 * SUM_OF)].flatten().astype(int)),
          f' | sum of all E:', np.sum(Y_scaler.inverse_transform(Y_test).flatten().astype(int)))
    print(f'sum of day P:', np.sum(Y_scaler.inverse_transform(test_predictions)[:SUM_OF].flatten().astype(int)),
          f' | sum of week P:', np.sum(Y_scaler.inverse_transform(test_predictions)[:(SUM_OF * 7)].flatten().astype(int)),
          f' | sum of month P:', np.sum(Y_scaler.inverse_transform(test_predictions)[:(SUM_OF * 7 * 30)].flatten().astype(int)),
          f' | sum of all P:', np.sum(Y_scaler.inverse_transform(test_predictions).flatten().astype(int)))






	
	
	

	
 
 


run(user_name)
