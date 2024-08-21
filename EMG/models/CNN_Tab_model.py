import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten , Conv1D 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
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

CNN_tab = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1])  # Output layer
])

CNN_tab.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[MeanSquaredError()])
hist_cnn_tab = CNN_tab.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_mse  = CNN_tab.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test Mean Squared Error:", test_mse)

rmse = np.sqrt(test_mse)
print("Root Mean Squared Error:", rmse)

import matplotlib.pyplot as plt

plt.title("CNN_Tabular_model")
plt.plot(hist_cnn_tab.history['loss'], label='Training loss')
plt.plot(hist_cnn_tab.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()