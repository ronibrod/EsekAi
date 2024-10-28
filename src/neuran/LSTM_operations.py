from .LSTM import model
from .db.model_weights import insert_model_data, save_new_model_data

def predict_model(user_name, predict_name, input_data):
    insert_model_data(user_name, predict_name)

    return model.predict(input_data)

def train_model(user_name, input_data, output_data):
    # print(input_data[:5])
    # print(output_data[:5])
    print(input_data.shape, output_data.shape)
    
    # insert_model_data(user_name)
  
    model.fit(input_data, output_data, epochs=20, batch_size=32)
  
    # save_new_model_data(user_name)
