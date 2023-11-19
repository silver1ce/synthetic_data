import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Generate some example data with categorical columns
# Replace this with your actual dataset
original_data = original_data

# Define the GAN-like generator and discriminator networks
def build_generator(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    embeddings = []
    for i in range(input_dim):
        emb = Embedding(output_dim=output_dim[i], input_dim=output_dim[i])(inputs[:, i:i+1])
        embeddings.append(emb)
    merged = Concatenate()(embeddings)
    output = Flatten()(merged)
    return Model(inputs, output)

def build_discriminator(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs, output)

# Set parameters
input_dim = len(original_data.columns)  # Number of categorical columns
output_dim = [len(original_data[column].unique()) for column in original_data.columns]

# Build and compile the generator and discriminator
generator = build_generator(input_dim, output_dim)
discriminator = build_discriminator(input_dim)

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False

gan_input = Input(shape=(input_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training loop
batch_size = 32
epochs = 10000

for epoch in range(epochs):
    noise = np.random.randint(0, np.max(output_dim), size=(batch_size, input_dim))
    generated_data = generator.predict(noise)
    real_data = original_data.sample(batch_size)
    x_combined = np.concatenate([real_data, generated_data])
    y_combined = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    d_loss = discriminator.train_on_batch(x_combined, y_combined)

    noise = np.random.randint(0, np.max(output_dim), size=(batch_size, input_dim))
    y_mislabeled = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, y_mislabeled)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Generate synthetic categorical data
num_samples = 1000
synthetic_data = generator.predict(np.random.randint(0, np.max(output_dim), size=(num_samples, input_dim)))
synthetic_data = pd.DataFrame(synthetic_data, columns=original_data.columns)
