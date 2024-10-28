import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from .const import time_steps


input_features = 20
output_features = 1

# Initialize the Sequential model
model = Sequential()

model.add(Input(shape=(input_features,)))  # input layer for 7 days, features
# model.add(Input(shape=(time_steps, input_features)))  # input layer for 7 days, features
# model.add(Dense(32, activation='relu', input_shape=(input_features,)))
# model.add(LSTM(units=9, input_shape=(time_steps, input_features), return_sequences=True))  # input layer for 7 days, features
# model.add(LSTM(9, return_sequences=True))  # First LSTM layer
# model.add(LSTM(9, return_sequences=False))  # Second LSTM layer
# model.add(Dense(128, activation='relu'))
# model.add(Dense(3, activation='relu'))  # First Dense layer
# model.add(Dense(3, activation='relu'))  # First Dense layer
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Conv1D(9, kernel_size=3, activation='relu'))
# model.add(tf.keras.layers.Conv1D(9, kernel_size=3, activation='relu'))
# model.add(tf.keras.layers.Flatten())
model.add(Dense(8, activation='relu'))  # First Dense layer
model.add(Dense(11, activation='relu'))  # Second Dense layer
# model.add(Dense(8, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(Dense(6, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(10, activation='relu'))
# model.add(Dense(units=10, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Dense(units=10, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(Dense(units=11, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(Dense(units=9, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# model.add(Dense(units=8, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(8, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(Dense(20, activation='relu'))


model.add(Dense(1, activation='linear'))

# with tf.variable_creator_scope('input'):
#   X = tf.placeholder(tf.int32, shape=(None, input_features))
# model.add(Input(shape=(input_features,)))
# model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
# model.add(Reshape((1, 128)))  # Reshape to (batch_size, time_steps, features)

# # Add an LSTM layer for sequence data
# model.add(LSTM(64, activation='relu'))
# model.add(Dense(1, activation='linear', kernel_initializer=HeNormal()))

# optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mean_squared_error')

# Compile the model
mse = 'mean_squared_error' # 0.715
mae = 'mean_absolute_error'
mape = 'mean_absolute_percentage_error'
msle = 'mean_squared_logarithmic_error' # 0.726
hu = 'huber' # 0.707
cc = 'categorical_crossentropy'
scc = 'sparse_categorical_crossentropy'
bc = 'binary_crossentropy' # 0.693

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
adam = tf.keras.optimizers.Adam(learning_rate=0.001) # 0.713
RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001)
adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.01)
adadelta = tf.keras.optimizers.Adadelta(learning_rate=1.0)
adamax = tf.keras.optimizers.Adamax(learning_rate=0.002)
nadam = tf.keras.optimizers.Nadam(learning_rate=0.002) # 0.714
ftrl = tf.keras.optimizers.Ftrl(learning_rate=0.001)

model.compile(optimizer='adam', loss='mean_squared_error')
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Display the model summary
# model.summary()
