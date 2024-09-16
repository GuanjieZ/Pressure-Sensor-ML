import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
necessary_file_path = "Q:\\mTPAD\\Pressure-Sensor-ML\\necessary_labled.txt"
carbon_file_path = "Q:\\mTPAD\\Pressure-Sensor-ML\\carbon_labeled.txt"

# Load the datasets
necessary_data = pd.read_csv(necessary_file_path, delimiter='\t')
carbon_data = pd.read_csv(carbon_file_path, delimiter='\t')

# Plot the necessary labeled data
plt.figure(figsize=(12, 6))

# Plot for necessary
plt.subplot(1, 2, 1)
plt.plot(necessary_data['Time'], necessary_data['Current'], label="Necessary", color='blue')
plt.title('Necessary Labeled Data')
plt.xlabel('Time')
plt.ylabel('Current')
plt.grid(True)

# Plot for carbon
plt.subplot(1, 2, 2)
plt.plot(carbon_data['Time'], carbon_data['Current'], label="Carbon", color='green')
plt.title('Carbon Labeled Data')
plt.xlabel('Time')
plt.ylabel('Current')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
