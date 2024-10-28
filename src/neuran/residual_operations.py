from .residuals_model import residuals_model
from .db.model_weights import insert_model_data, get_scaler

def predict_residuals_model(user_name, input_data):
    predict_name = '_residual_nyc'
    scaler_name = 'residuals_scaler_nyc'
    insert_model_data(user_name, predict_name)
    
    predict_residuals = residuals_model.predict(input_data)
    residuals_scaler = get_scaler(user_name, scaler_name)
    inverse_predict_residuals = residuals_scaler.inverse_transform(predict_residuals)
    return inverse_predict_residuals
