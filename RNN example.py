import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load the data
data = pd.read_csv('Q:\\mTPAD\\Pressure-Sensor-ML\\necessary_labled.txt', delimiter='\t')

# Extract 'Current' column and replace NaNs with 0
current_values = data['Current'].fillna(0).values

# Create labels for sequences: 1 if in a continuous non-zero block, else 0
labels = np.zeros_like(current_values)
non_zero_indices = np.where(current_values > 0)[0]

# Label continuous non-zero segments as 1 (word 'necessary')
start = None
for i in range(len(current_values)):
    if current_values[i] > 0:
        if start is None:
            start = i  # Start of a non-zero sequence
        labels[start:i + 1] = 1  # Label the continuous sequence
    else:
        start = None

# Reshape into sequences (windows) for RNN input, e.g., window length of 10
sequence_length = 10
X_sequences = []
y_sequences = []

for i in range(len(current_values) - sequence_length):
    X_sequences.append(current_values[i: i + sequence_length])
    # If any of the points in the sequence contains the word, label it as 1
    y_sequences.append(int(np.any(labels[i: i + sequence_length])))

X_sequences = np.array(X_sequences).reshape(-1, sequence_length, 1)
y_sequences = np.array(y_sequences)

# Split into training and test sets
split_idx = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(64, input_shape=(sequence_length, 1), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
