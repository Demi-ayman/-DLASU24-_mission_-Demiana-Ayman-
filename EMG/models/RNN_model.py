import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data
X_train_padding = np.load("X_train_padding.npy").astype(np.float32)
y_train_padding = np.load("y_train_padding.npy").astype(np.float32)

# Split data into training, validation, and test sets 
X_train, X_temp, y_train, y_temp = train_test_split(X_train_padding, y_train_padding, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the model
model = Sequential()

# LSTM layer
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))
model.add(BatchNormalization())

# GRU layer (You can replace this with another LSTM if preferred)
model.add(GRU(128, activation='relu', return_sequences=False))
model.add(Dropout(0.5))
model.add(BatchNormalization())

# Fully Connected Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(y_train.shape[-1], activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Print model summary
model.summary()

# Modify y_train and y_val to match the output shape (if necessary)
y_train = y_train[:, -1, :]  # Use only the last timestep
y_val = y_val[:, -1, :]

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Plotting Training & Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
