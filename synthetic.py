import numpy as np
import pandas as pd
import random
import faker  # A library for generating fake data (e.g., addresses)
from datetime import datetime, timedelta
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Define a function to create a GAN for generating numerical data
def create_gan(input_dim, output_dim):
    # Generator
    generator_input = Input(shape=(input_dim,))
    generator_output = Dense(128, activation='relu')(generator_input)
    generator_output = Dense(output_dim, activation='linear')(generator_output)
    generator = Model(generator_input, generator_output)

    # Discriminator
    discriminator_input = Input(shape=(output_dim,))
    discriminator_output = Dense(128, activation='relu')(discriminator_input)
    discriminator_output = Dense(1, activation='sigmoid')(discriminator_output)
    discriminator = Model(discriminator_input, discriminator_output)
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # GAN
    gan_input = Input(shape=(input_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return generator, discriminator, gan

# Define a function to generate synthetic numerical data using GAN
def generate_numerical_data(generator, num_samples=1000):
    noise = np.random.randn(num_samples, generator.input_shape[1])
    synthetic_data = generator.predict(noise)
    return synthetic_data

# Define a function to generate synthetic data including text using Faker
def generate_synthetic_data_with_text(original_data, num_samples=1000):
    # Create an empty synthetic dataframe with the same columns as the original data
    synthetic_data = pd.DataFrame(columns=original_data.columns)

    # Create a Faker object for generating fake data (e.g., addresses)
    fake = faker.Faker()

    for _ in range(num_samples):
        # Create a dictionary to store synthetic data for one row
        synthetic_row = {}

        for column in original_data.columns:
            data_type = original_data[column].dtype

            if np.issubdtype(data_type, np.number):
                # If the column is numeric, generate a random number within a reasonable range
                min_value = original_data[column].min()
                max_value = original_data[column].max()
                synthetic_value = np.random.uniform(min_value, max_value)
            elif np.issubdtype(data_type, np.datetime64):
                # If the column is a datetime, generate a random date within a reasonable range
                min_date = original_data[column].min()
                max_date = original_data[column].max()
                synthetic_date = min_date + (max_date - min_date) * random.random()
                synthetic_value = datetime.fromordinal(int(synthetic_date))
            elif data_type == 'object':
                # If the column is an object, generate synthetic data based on column name
                if 'company' in column.lower():
                    synthetic_value = fake.company()
                elif 'address' in column.lower():
                    synthetic_value = fake.address()
                elif 'customer' in column.lower():
                    synthetic_value = fake.name()
                else:
                    # If the column doesn't match known patterns, generate a random word
                    synthetic_value = fake.word()

            synthetic_row[column] = synthetic_value

        # Append the synthetic row to the synthetic data
        synthetic_data = synthetic_data.append(synthetic_row, ignore_index=True)

    return synthetic_data

# Example usage
if __name__ == "__main__":
    # Sample numerical data dimensions (customize for your data)
    num_samples = 1000
    num_features = 5

    # Generate sample numerical data using GAN
    input_dim = 100  # Size of the random noise vector for the generator
    generator, _, _ = create_gan(input_dim, num_features)
    numerical_data = generate_numerical_data(generator, num_samples=num_samples)

    # Load your original dataset (replace with your data loading code)
    original_data = pd.read_csv("your_data.csv")

    # Generate synthetic data including text using Faker
    synthetic_data = generate_synthetic_data_with_text(original_data, num_samples=num_samples)

   