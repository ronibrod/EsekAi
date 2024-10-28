
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# יצירת המודל
model = Sequential()

# הוספת שכבת הקלט (Dense Layer) עם 32 נוירונים ופונקציית הפעלה relu
model.add(Dense(32, activation='relu', input_shape=(2,)))

# הוספת שכבת הפלט עם נוירון אחד
model.add(Dense(1))

# קומפילציה של המודל
model.compile(optimizer='adam', loss='mean_squared_error')

# הדפסת סיכום המודל
model.summary()

import numpy as np

# יצירת נתוני אימון (דוגמה)
X_train = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]])
y_train = np.array([0.3, 0.5, 0.7, 0.9, 1.1])

# אימון המודל
model.fit(X_train, y_train, epochs=100, batch_size=1)

# ניבוי באמצעות המודל
predictions = model.predict(X_train)
print(predictions)
