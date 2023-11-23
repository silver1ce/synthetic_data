import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Load and preprocess the example dataset (assuming it's in CSV format)
example_file = 'example.csv'
data = pd.read_csv(example_file)
# Preprocess the data as needed (e.g., scaling, normalization)

# Convert the preprocessed data to a PyTorch tensor
data_tensor = torch.tensor(data.values, dtype=torch.float32)

# Define a Variational Autoencoder (VAE) model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Define the encoder and decoder layers
        # Customize the architecture as needed based on your data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # The last layer outputs mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Sigmoid for bounded data (0 to 1)
        )

    def reparameterize(self, mu, log_var):
        # Reparameterization trick to sample from the latent space
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Forward pass through the encoder and decoder
        enc = self.encoder(x)
        mu, log_var = enc[:, :latent_dim], enc[:, latent_dim:]
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

# Define hyperparameters
input_dim = data_tensor.shape[1]
latent_dim = 10
batch_size = 64
epochs = 100
learning_rate = 0.001

# Create a DataLoader for the dataset
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the VAE model and optimizer
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch_data in dataloader:
        optimizer.zero_grad()
        recon_data, mu, log_var = model(batch_data[0])
        # Define a suitable reconstruction loss (e.g., mean squared error)
        reconstruction_loss = nn.MSELoss()(recon_data, batch_data[0])
        # Define a suitable regularization loss term (e.g., KL divergence)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = reconstruction_loss + kl_divergence
        total_loss.backward()
        optimizer.step()

# Generate synthetic data samples
num_samples = 1000
with torch.no_grad():
    z_samples = torch.randn(num_samples, latent_dim)
    synthetic_samples = model.decoder(z_samples).numpy()

# Create a DataFrame for the synthetic data
synthetic_df = pd.DataFrame(synthetic_samples, columns=data.columns)

# Save the synthetic data to a CSV file
synthetic_file = 'synthetic_data.csv'
synthetic_df.to_csv(synthetic_file, index=False)

print(f"Synthetic data saved to {synthetic_file}")
