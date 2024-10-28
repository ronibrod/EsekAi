import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal


input_features = 20
output_features = 1

residuals_model = Sequential()

residuals_model.add(Input(shape=(input_features,)))
residuals_model.add(Dense(10, activation='relu'))
residuals_model.add(Dense(11, activation='relu'))
# residuals_model.add(tf.keras.layers.Dropout(0.5))
# residuals_model.add(Dense(11, activation='relu'))
residuals_model.add(Dense(10, activation='relu'))
residuals_model.add(Dense(1, activation='linear'))

residuals_model.compile(optimizer='adam', loss='mean_squared_error')
