import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K

# Sample original dataset with an object column
# Replace this with your actual data loading code
original_data = pd.DataFrame({'object_col': ['Category A', 'Category B', 'Category C'] * 333})

# Define a VAE architecture for generating object data
def build_object_vae(input_dim, latent_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(256, activation='relu')(input_layer)
    encoded = Dense(128, activation='relu')(encoded)
    
    # Latent space mean and log variance
    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)
    
    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling)([z_mean, z_log_var])
    
    # Decoder
    decoded = Dense(128, activation='relu')(z)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='softmax')(decoded)  # Softmax activation for object data
    
    # VAE model
    vae = Model(input_layer, decoded)
    
    # VAE loss function
    reconstruction_loss = mse(input_layer, decoded)
    reconstruction_loss *= input_dim  # Adjust reconstruction loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return vae

# Define a function to generate synthetic object data
def generate_synthetic_object_data(vae, num_samples=1000):
    latent_dim = vae.layers[5].output_shape[1]  # Get the latent dimension
    latent_samples = np.random.normal(0, 1, (num_samples, latent_dim))
    synthetic_data = vae.predict(latent_samples)
    return synthetic_data

# Example usage
if __name__ == "__main__":
    num_samples = 1000  # Number of synthetic samples to generate
    
    # Build and train the object VAE
    object_input_dim = len(original_data['object_col'].unique())  # Dimension for one-hot encoding
    object_vae = build_object_vae(object_input_dim, latent_dim=10)
    
    # Train the object VAE on object data (replace with your data loading and preprocessing)
    object_real_data = original_data['object_col'].values
    
    # One-hot encode the object data
    object_real_data_encoded = pd.get_dummies(object_real_data)
    
    # Training code for object VAE goes here
    
    # Generate synthetic object data
    synthetic_object_data = generate_synthetic_object_data(object_vae, num_samples=num_samples)

    # Now, 'synthetic_object_data' contains synthetic object data.
