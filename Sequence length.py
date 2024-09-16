import pandas as pd
import matplotlib.pyplot as plt

# Load the file
file_path = 'Q:\\mTPAD\\Pressure-Sensor-ML\\necessary_labled_raw.txt'

# Load the data into a DataFrame
data = pd.read_csv(file_path, sep="\t", header=None, names=["Time", "Current"])

# Convert the 'Time' column to numeric to handle subtraction correctly
data["Time"] = pd.to_numeric(data["Time"], errors='coerce')

# Identify the sequences of non-NaN values
sequences = []
current_sequence = []
start_sequence = 0
time_stamp = []

for i in range(len(data["Current"])-1):
    if pd.isna(data["Current"][i+1]) == False:
        start_sequence = 1
        current_sequence.append(data["Current"][i+1])
    elif pd.isna(data["Current"][i+1]) == True and start_sequence == 1:
        start_sequence = 0
        sequences.append(current_sequence)
        time_stamp.append(data["Time"][i+1])
        current_sequence = []

# Save sequences list to a text file
with open('Q:\\mTPAD\\Pressure-Sensor-ML\\necessary_labled_pure.txt', 'w') as f:
    for seq in sequences:
        f.write("%s\n" % seq)


length = []        
for i in range(len(sequences)):
    length.append(len(sequences[i]))
    if len(sequences[i]) < 30:
        print(sequences[i])
        print(time_stamp[i])
        print()

print(max(length))
print(min(length))