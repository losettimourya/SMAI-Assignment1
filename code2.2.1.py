import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from the .npy file
data = np.load('data.npy', allow_pickle=True)

# Extract the labels (assuming labels are in the 4th column)
labels = data[:, 3]

# Calculate the frequency of each label using NumPy's unique function
unique_labels, label_counts = np.unique(labels, return_counts=True)

# Create a bar graph to visualize the label distribution
plt.figure(figsize=(10, 6))
plt.bar(unique_labels, label_counts, align='center', alpha=0.7)
plt.xlabel('Label Name')
plt.ylabel('Frequency')
plt.title('Label Distribution in the Dataset')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

# Show the plot
plt.show()
