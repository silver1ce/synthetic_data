import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Load the sample CSV dataset (replace 'sample.csv' with your dataset file)
df = pd.read_csv('sample.csv')

# Preprocess the data (handle missing values, encode categorical variables, normalize numeric data)
# You can use libraries like scikit-learn for preprocessing
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Define the GAN architecture
def build_generator(latent_dim, num_features):
    input_noise = Input(shape=(latent_dim,))
    x = Dense(128, activation='relu')(input_noise)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_features, activation='sigmoid')(x)
    generator = Model(inputs=input_noise, outputs=x)
    return generator

def build_discriminator(num_features):
    input_data = Input(shape=(num_features,))
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(input_data)
    x = Dense(128, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inputs=input_data, outputs=x)
    return discriminator

def build_gan(generator, discriminator):
    discriminator.trainable = False
    input_noise = Input(shape=(latent_dim,))
    generated_data = generator(input_noise)
    validity = discriminator(generated_data)
    gan = Model(inputs=input_noise, outputs=validity)
    return gan

# Hyperparameters (adjust as needed)
latent_dim = 32
num_features = df.shape[1]  # Number of columns in your dataset
batch_size = 64
epochs = 10000

# Build and compile the models
generator = build_generator(latent_dim, num_features)
discriminator = build_discriminator(num_features)
gan = build_gan(generator, discriminator)

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))

# Training loop
for epoch in range(epochs):
    # Sample random noise as input
    noise = np.random.randn(batch_size, latent_dim)
    
    # Generate synthetic data
    synthetic_data = generator.predict(noise)
    
    # Train discriminator on real data
    real_labels = np.ones((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(df.sample(batch_size).values, real_labels)
    
    # Train discriminator on generated data
    fake_labels = np.zeros((batch_size, 1))
    d_loss_fake = discriminator.train_on_batch(synthetic_data, fake_labels)
    
    # Calculate discriminator loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train generator
    noise = np.random.randn(batch_size, latent_dim)
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Generate synthetic data
num_synthetic_samples = 1000
synthetic_noise = np.random.randn(num_synthetic_samples, latent_dim)
synthetic_data = generator.predict(synthetic_noise)

# Convert synthetic data to a DataFrame (assuming column names are preserved)
synthetic_df = pd.DataFrame(data=synthetic_data, columns=df.columns)

# You can now use 'synthetic_df' as your synthetic dataset
