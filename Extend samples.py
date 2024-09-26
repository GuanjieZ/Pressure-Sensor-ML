import pandas as pd
import matplotlib.pyplot as plt
import random

# Load the file
file_path = 'D:\\CAD\\Pressure-Sensor-ML\\experiment_labled_raw.txt'
noise_path = 'D:\\CAD\\Pressure-Sensor-ML\\blank.txt'

# Load the data into a DataFrame
data = pd.read_csv(file_path, sep="\t", header=None, names=["Time", "Current"])
noise = pd.read_csv(noise_path, sep="\t", header=None, names=["Time", "Current"])

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


for i in range(len(sequences)):
    while len(sequences[i]) < 150:
        sequences[i].append(random.choice(noise["Current"]))

length = []        
for i in range(len(sequences)):
    length.append(len(sequences[i]))
    if len(sequences[i]) < 30:
        print(sequences[i])
        print(time_stamp[i])
        print()

print(len(length))

print(max(length))
print(min(length))


# # Plotting each sequence separately
# plt.figure()
# plt.plot([float(x) for x in sequences[1]])
# plt.show()



# # Save sequences list to a text file
# with open('D:\\CAD\\Pressure-Sensor-ML\\magic_labled_extended.txt', 'w') as f:
#     for seq in sequences:
#         f.write("%s\n" % seq)