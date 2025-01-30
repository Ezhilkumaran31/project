import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load encoded features
X_train_encoded = np.load('X_train_encoded.npy')
X_test_encoded = np.load('X_test_encoded.npy')

# Load labels from feature.py
from feature import y_train, y_test  # Ensure feature.py contains y_train, y_test

# Reshape for RNN (samples, timesteps, features)
X_train_rnn = X_train_encoded.reshape((X_train_encoded.shape[0], 1, X_train_encoded.shape[1]))
X_test_rnn = X_test_encoded.reshape((X_test_encoded.shape[0], 1, X_test_encoded.shape[1]))

# Build RNN Model
model = Sequential([
    SimpleRNN(64, activation='relu', return_sequences=True, input_shape=(1, X_train_encoded.shape[1])),
    SimpleRNN(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the RNN model
model.fit(X_train_rnn, y_train, epochs=50, batch_size=32, validation_data=(X_test_rnn, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test_rnn, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
