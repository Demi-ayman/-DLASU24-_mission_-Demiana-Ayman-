import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Reshape, Attention, Input, Activation
from tensorflow.keras import Model
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
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# CNN Layers
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
pool1 = MaxPooling1D(pool_size=2)(conv1)
bn1 = BatchNormalization()(pool1)

conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(bn1)
pool2 = MaxPooling1D(pool_size=2)(conv2)
bn2 = BatchNormalization()(pool2)

conv3 = Conv1D(filters=256, kernel_size=3, activation='relu')(bn2)
pool3 = MaxPooling1D(pool_size=2)(conv3)
bn3 = BatchNormalization()(pool3)

# LSTM Layer
lstm_out = LSTM(64, return_sequences=True)(bn3)

# Attention Layer
attention = Attention()([lstm_out, lstm_out])
attention = Dense(64, activation='relu')(attention)

# Flatten the attention output and add Dense layers
flat = Reshape((attention.shape[1] * attention.shape[2],))(attention)
dense1 = Dense(128, activation='relu')(flat)
dropout1 = Dropout(0.5)(dense1)

dense2 = Dense(64, activation='relu')(dropout1)
dropout2 = Dropout(0.5)(dense2)

# Output Layer
output_layer = Dense(y_train.shape[-1], activation='linear')(dropout2)

# Create and compile model
model = Model(inputs=input_layer, outputs=output_layer)
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