import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from ..LSTM import model
from .index import get_model_weights_collection

def insert_model_data(user_name, model_name=''):
    model_weights_collection = get_model_weights_collection(user_name)
    
    model_data = model_weights_collection.find_one({'model_name': f'LSTM_model{model_name}'})
    if model_data:
        weights_data = model_data['weights']
        model.set_weights(pickle.loads(weights_data))

def save_new_model_data(user_name, model_name=''):
    model_weights_collection = get_model_weights_collection(user_name)
  
    weights_data = pickle.dumps(model.get_weights())
    filter_query = {'model_name': f'LSTM_model{model_name}'}
    update_query = {'$set': {'weights': weights_data}}
    model_weights_collection.update_one(filter_query, update_query, upsert=True)
    
def get_scaled_input_data(user_name, model_name, input_data):
    model_weights_collection = get_model_weights_collection(user_name)
    scaler_params = model_weights_collection.find_one({'model_name': model_name})

    if scaler_params is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        input_scaler = scaler.fit(input_data)
        
        scaler_params = {
            'min': input_scaler.data_min_.tolist(),
            'max': input_scaler.data_max_.tolist(),
            'scale': input_scaler.scale_.tolist(),
            'min_range': input_scaler.min_.tolist(),
            'model_name': model_name
        }
        model_weights_collection.insert_one(scaler_params)
        
        return input_scaler.transform(input_data)
    
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.data_min_ = np.array(scaler_params['min'])
        scaler.data_max_ = np.array(scaler_params['max'])
        scaler.scale_ = np.array(scaler_params['scale'])
        scaler.min_ = np.array(scaler_params['min_range'])
        
        return scaler.transform(input_data)

def get_scaler(user_name, model_name):
    model_weights_collection = get_model_weights_collection(user_name)
    scaler_params = model_weights_collection.find_one({'model_name': model_name})
    
    if scaler_params is None:
        raise ValueError(f"Scaler parameters for model '{model_name}' not found.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_ = np.array(scaler_params['min'])
    scaler.data_max_ = np.array(scaler_params['max'])
    scaler.scale_ = np.array(scaler_params['scale'])
    scaler.min_ = np.array(scaler_params['min_range'])

    return scaler
