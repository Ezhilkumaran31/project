import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)  # Extract encoded features
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

# Load preprocessed features from feature.py
from feature import X_train, X_test  # Ensure feature.py contains X_train, X_test

input_dim = X_train.shape[1]
autoencoder, encoder = build_autoencoder(input_dim)

# Train the AutoEncoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Extract features using Encoder
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Save encoded features
np.save('X_train_encoded.npy', X_train_encoded)
np.save('X_test_encoded.npy', X_test_encoded)
