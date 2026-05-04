# Variational Autoencoder (VAE)

import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 🔹 Parameters
input_dim = 20
latent_dim = 2

# 🔹 Encoder
inputs = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(inputs)
latent = Dense(latent_dim)(encoded)

# 🔹 Decoder
decoded = Dense(10, activation='relu')(latent)
outputs = Dense(input_dim, activation='sigmoid')(decoded)

# 🔹 VAE Model
vae = Model(inputs, outputs)
vae.compile(optimizer='adam', loss='mse')

# 🔹 Dummy Data
X_train = np.random.rand(1000, input_dim)

# 🔹 Train
vae.fit(X_train, X_train, epochs=10, batch_size=32)

# 🔹 Test Reconstruction
sample = np.random.rand(1, input_dim)
reconstructed = vae.predict(sample)

print("Original:", sample)
print("Reconstructed:", reconstructed)