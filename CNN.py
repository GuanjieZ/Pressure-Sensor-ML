import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def load_data(filename):
    sequences = []
    with open(filename, 'r') as file:
        for line in file:
            # Clean the line and remove any extra characters
            line = line.strip().strip('[]').replace("'", "").replace(" ", "")
            
            # Convert the cleaned line into a list of floats
            sequence = list(map(float, line.split(',')))
            sequences.append(sequence)
    return sequences

# Load necessary and carbon sequences
necessary_sequences = load_data('Q:\\mTPAD\\Pressure-Sensor-ML\\necessary_labled_extended.txt')
carbon_sequences = load_data('Q:\\mTPAD\\Pressure-Sensor-ML\\carbon_labled_extended.txt')


# Label the sequences
necessary_labels = np.zeros(len(necessary_sequences))
carbon_labels = np.ones(len(carbon_sequences))

# Combine the sequences and labels
all_sequences = necessary_sequences + carbon_sequences
all_labels = np.concatenate([necessary_labels, carbon_labels])

# Pad sequences to have the same length
max_seq_length = max([len(seq) for seq in all_sequences])
X = pad_sequences(all_sequences, maxlen=max_seq_length, dtype='float32')

# Convert labels to categorical
y = to_categorical(all_labels, num_classes=2)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data to be compatible with RNN (samples, time steps, features)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Create a Sequential CNN model
model = Sequential()

# Add a 1D convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))

# Add a MaxPooling layer to downsample the sequence
model.add(MaxPooling1D(pool_size=2))

# Add a second Conv1D layer (optional, can experiment with this)
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))

# Add another MaxPooling layer (optional, can experiment with this)
model.add(MaxPooling1D(pool_size=2))

# Flatten the output of the convolutional layers to feed into Dense layers
model.add(Flatten())

# Add a Dense layer with dropout for regularization
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))

# Output layer for binary classification
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=110, batch_size=256, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')


# Generate predictions for the test set
predictions = model.predict(X_test)

# Let's plot the first 10 sequences from the test data
fig, axes = plt.subplots(10, 1, figsize=(10, 20))

for i in range(10):
    sequence = X_test[i].squeeze()
    true_label = np.argmax(y_test[i])  # Convert one-hot to the original label (0 or 1)
    predicted_label = np.argmax(predictions[i])  # Get the predicted class
    
    axes[i].plot(sequence)
    axes[i].set_title(f'Sequence {i+1} - True: {true_label}, Predicted: {predicted_label}')
    axes[i].set_xlabel('Time Steps')
    axes[i].set_ylabel('Value')

plt.tight_layout()
plt.show()
