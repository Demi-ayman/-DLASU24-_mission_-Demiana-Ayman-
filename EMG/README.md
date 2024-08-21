DNN_Model:
  Loading Data: Import data from files into X_train_tabular and y_train_tabular.
  Splitting Data: Divide data into training, validation, and test sets.
  Scaling: Normalize feature values to improve model performance.
  Input Layer: 64 neurons with ReLU activation. The input shape matches the number of features in X_train.
  Hidden Layers: 32 neurons with ReLU activation.
  Output Layer: Number of neurons equals the number of output variables in y_train.
  Compile: Configures the model with loss function and optimizer.
  Train: Fits the model to the training data and validates using the validation set
  Evaluation: Tests the model’s performance on unseen data.
  Metrics: Outputs loss and Mean Squared Error (MSE). RMSE is computed as the square root of MSE.
  Plot: Shows how the loss changes over training epochs, indicating model performance and convergence.

CNN_Tabular:
  Loading Data: Load data from files into arrays X_train_tabular and y_train_tabular.
  Splitting Data: Divide the data into training, validation, and test sets for model evaluation.
  Scaling: Normalize the feature values to have zero mean and unit variance, which helps in stabilizing and speeding up the training process.
  1D Convolutional Layer: Applies 32 filters of size 3 to the input data. This layer detects local patterns or features.
  Flatten Layer: Converts the 2D output from the convolutional layer into a 1D vector, suitable for Dense layers.
  Dense Layers: Fully connected layers for further processing and output. The final layer's size matches the number of target variables.
  Compile: Set up the model with the chosen optimizer (Adam), loss function (mean_squared_error), and evaluation metric (MeanSquaredError).
  Train: Fit the model to the training data, and validate on the validation data.
  Evaluation: Test the model on the unseen test set to measure performance.
  Metrics: Report loss and Mean Squared Error, and compute Root Mean Squared Error (RMSE) for interpretability.
  Plot: Graphs showing how the loss changes during training and validation. This helps in understanding if the model is overfitting, underfitting, or learning effectively.
CNN_Sequential:
  Loading Data: Load the data from .npy files and convert it to float32.
  Splitting Data: Divide the data into training, validation, and test sets for model evaluation.
  1D Convolutional Layers: Apply convolutional operations to detect patterns in the time series data. Each convolutional layer is followed by MaxPooling to reduce dimensionality and BatchNormalization to stabilize training.
  1st Conv1D Layer: 64 filters with a kernel size of 3.
  2nd Conv1D Layer: 128 filters with a kernel size of 3.
  3rd Conv1D Layer: 256 filters with a kernel size of 3.
  Flatten Layer: Converts the 3D output from the convolutional layers into a 1D vector.
  Dense Layers: Fully connected layers for further processing.
  Dropout: Regularization technique to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
  Compile: Set up the model with the Adam optimizer, Mean Squared Error loss function, and Mean Absolute Error metric.
  Model Summary: Print a summary of the model architecture.
  Adjust Labels: Modify the target labels to match the output shape, focusing on the last timestep.
  Training: Fit the model on the training data and validate it using the validation set.
  Plot Loss Curves: Visualize the training and validation loss over epochs to understand how well the model is learning and if it is overfitting or underfitting.
RNN_model:
  Loading Data: Data is loaded from .npy files and converted to float32 for processing.
  Splitting Data: The dataset is divided into training, validation, and test sets.
  LSTM Layer:

  LSTM(64): 64 units, return_sequences=True means it returns the full sequence to the next layer.
  Dropout(0.5): Regularization to prevent overfitting by randomly setting 50% of input units to 0.
  BatchNormalization(): Normalizes activations to stabilize and accelerate training.
  GRU Layer:
  
  GRU(128): 128 units, return_sequences=False means it returns only the output of the last timestep.
  Dropout(0.5): Regularization similar to the LSTM layer.
  BatchNormalization(): Normalizes activations.
  Fully Connected Layers:
  
  Dense(128) and Dense(64): Fully connected layers for further processing.
  Dropout(0.5): Regularization for dense layers.
  Output Layer:
  
  Dense(y_train.shape[-1], activation='linear'): Output layer with the same number of units as the last dimension of y_train, typically for regression tasks.
  Compile: Configure the model with the Adam optimizer and Mean Squared Error loss function. Mean Absolute Error is also tracked.
  Model Summary: Display the model architecture.
  Adjust Labels: If necessary, ensure the target labels y_train and y_val have the correct shape by selecting the last timestep.
  Training: Train the model with the training data and validate it using the validation data.
  Plot Loss Curves: Plot training and validation loss over epochs to visualize how the model's performance evolves during training. This helps in assessing whether the model is overfitting, underfitting, or learning appropriately.
CNN_LSTM:
  Loading Data: Data is loaded from .npy files and converted to float32.
  Splitting Data: The dataset is split into training, validation, and test sets.
  Convolutional Layers:

  Conv1D(64, kernel_size=3): Applies 64 filters of size 3 to the input sequence, extracting features.
  
  MaxPooling1D(pool_size=2): Reduces the dimensionality by taking the maximum value over a window of size 2.
  
  BatchNormalization(): Normalizes the output of the convolutional layers to stabilize training.
  
  Conv1D(128, kernel_size=3): Applies 128 filters to capture more complex patterns.
  
  MaxPooling1D(pool_size=2): Further reduces dimensionality.
  
  Reshape Layer:
  
  Reshape(): Converts the output from the convolutional layers to a shape suitable for the LSTM layer. Here, model.output_shape is used to get the shape of the output from the previous layer. The target shape for LSTM is (time_steps, features).
  LSTM Layer:
  
  LSTM(64): 64 units, processes the sequential data, return_sequences=False means only the output of the last timestep is returned.
  Dropout(0.5): Regularization to reduce overfitting.
  BatchNormalization(): Normalizes the output of the LSTM layer.
  Fully Connected Layers:
  
  Dense(128) and Dense(64): Fully connected layers to process features further.
  Dropout(0.5): Regularization to reduce overfitting.
  Output Layer:
  
  Dense(y_train.shape[-1], activation='linear'): Output layer with units matching the number of outputs, typically used for regression tasks.
  Compile: Configure the model with the Adam optimizer and Mean Squared Error loss function. Mean Absolute Error is also tracked.
  Model Summary: Display the model architecture.
  Adjust Labels: Ensure target labels y_train and y_val have the correct shape by selecting the last timestep.
  Train the Model: Train the model using the training data and validate it with the validation data.
  Plot Loss Curves: Visualizes the loss curves for training and validation, allowing you to assess model performance over epochs.

Attention_model:
  Loading Data: Load and convert data to float32.
  Splitting Data: Divide data into training, validation, and test sets.
  Input Layer:

  Input(shape=(X_train.shape[1], X_train.shape[2])): Defines the shape of the input data.
  CNN Layers:
  
  Conv1D(filters=64, kernel_size=3): 1D convolution with 64 filters and a kernel size of 3.
  
  MaxPooling1D(pool_size=2): Reduces dimensionality by pooling.
  
  BatchNormalization(): Normalizes activations for stable training.
  
  Conv1D(filters=128, kernel_size=3): Adds more convolutional filters.
  
  MaxPooling1D(pool_size=2): Additional pooling.
  
  BatchNormalization(): Further normalization.
  
  Conv1D(filters=256, kernel_size=3): More filters for deeper feature extraction.
  
  MaxPooling1D(pool_size=2): Reduces dimensionality.
  
  BatchNormalization(): Final normalization in the CNN section.
  
  LSTM Layer:
  
  LSTM(64, return_sequences=True): Captures temporal dependencies, return_sequences=True means it outputs sequences for attention.
  Attention Layer:
  
  Attention(): Applies attention mechanism to LSTM output. Attention helps the model focus on important parts of the sequence.
  Dense(64, activation='relu'): Processes the attention output.
  Flatten and Dense Layers:
  
  Reshape(): Reshapes attention output to a flat vector.
  Dense(128) and Dense(64): Fully connected layers to process features.
  Dropout(0.5): Regularization to prevent overfitting.
  Output Layer:
  
  Dense(y_train.shape[-1], activation='linear'): Output layer with linear activation for regression tasks.
  Compile: Configures the model with the Adam optimizer and Mean Squared Error loss function. Mean Absolute Error is also tracked.
  Model Summary: Displays the model architecture.
  Adjust Labels: Ensure target labels y_train and y_val have the correct shape by selecting the last timestep.
  Train the Model: Fit the model with training data and validate with validation data.
  Plot Loss Curves: Visualizes the loss curves for training and validation, providing insights into model performance over epochs.
  
