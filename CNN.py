import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# Set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Function to load data from a file
def load_data(filename):
    sequences = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().strip('[]').replace("'", "").replace(" ", "")
            sequence = list(map(float, line.split(',')))
            sequences.append(sequence)
    return sequences

# Load sequences for all classes
tsen1_sequences = load_data('D:\\CAD\\Pressure-Sensor-ML\\necessary_labled_extended.txt')  # renamed from necessary
tsen2_sequences = load_data('D:\\CAD\\Pressure-Sensor-ML\\carbon_labled_extended.txt')     # renamed from carbon
magic_sequences = load_data('D:\\CAD\\Pressure-Sensor-ML\\magic_labled_extended.txt')
hello_sequences = load_data('D:\\CAD\\Pressure-Sensor-ML\\hello_labled_extended.txt')
information_sequences = load_data('D:\\CAD\\Pressure-Sensor-ML\\information_labled_extended.txt')
experiment_sequences = load_data('D:\\CAD\\Pressure-Sensor-ML\\experiment_labled_extended.txt')

# Label the sequences
tsen1_labels = np.zeros(len(tsen1_sequences))  # renamed from necessary_labels
tsen2_labels = np.ones(len(tsen2_sequences))   # renamed from carbon_labels
magic_labels = np.full(len(magic_sequences), 2)
hello_labels = np.full(len(hello_sequences), 3)
information_labels = np.full(len(information_sequences), 4)
experiment_labels = np.full(len(experiment_sequences), 5)

# Combine all sequences and labels
all_sequences = (tsen1_sequences + tsen2_sequences +
                 magic_sequences + hello_sequences + 
                 information_sequences + experiment_sequences)
all_labels = np.concatenate([tsen1_labels, tsen2_labels, 
                             magic_labels, hello_labels, 
                             information_labels, experiment_labels])

# Pad sequences
max_seq_length = max([len(seq) for seq in all_sequences])
X = pad_sequences(all_sequences, maxlen=max_seq_length, dtype='float32')

# Convert labels to categorical (6 classes)
y = to_categorical(all_labels, num_classes=6)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Create Sequential CNN model
model = Sequential()

# 1D Convolutional layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))

# Output layer for 6 classes
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model to ensure the input shape is known
model.fit(X_train, y_train, epochs=110, batch_size=256, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Generate predictions for the test set
predictions = model.predict(X_test)

# Convert predictions and true labels from one-hot encoding to class labels
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

# Class names (corresponding to the class indices)
class_names = ['tsen1', 'tsen2', 'magic', 'hello', 'information', 'experiment']

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix with class names
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45)
plt.show()

# Now that the model has been trained and called, we can create the intermediate model
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)

# Get features from the second-to-last Dense layer
X_test_features = intermediate_layer_model.predict(X_test)

# Apply t-SNE to reduce the dimensionality of the feature vectors to 2D
tsne = TSNE(n_components=2, random_state=42)
X_test_tsne = tsne.fit_transform(X_test_features)

# Plot t-SNE graph
plt.figure(figsize=(10, 8))
for i, label in enumerate(class_names):
    indices = np.where(y_true == i)
    plt.scatter(X_test_tsne[indices, 0], X_test_tsne[indices, 1], label=label, s=50)
    
plt.title('t-SNE Visualization of Test Data')
plt.legend()
plt.show()
