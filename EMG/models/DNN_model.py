import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

# Load the data
X_train_tabular = np.load("X_train_tabular.npy")
y_train_tabular = np.load("y_train_tabular.npy")

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_train_tabular, y_train_tabular, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

DNN = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dense(32, activation='relu'),
    Dense(y_train.shape[1])  # Output layer
])

# Compile the model
DNN.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[MeanSquaredError()])

# Train the model
hist_dnn = DNN.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_mse  = DNN.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test Mean Squared Error:", test_mse)

rmse = np.sqrt(test_mse)
print("Root Mean Squared Error:", rmse)

import matplotlib.pyplot as plt

plt.title("DNN_model")
plt.plot(hist_dnn.history['loss'], label='Training loss')
plt.plot(hist_dnn.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()