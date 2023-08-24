import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Example label combinations
all_label_combinations = [(0, 2), (1,), (0, 1, 2)]  # Replace with your label combinations

# Example y_train and y_val
y_train = [(0, 2), (1, 2), (0,), (0, 1), (2,)]
y_val = [(1,), (0, 2), (0, 1), (1, 2), (0,)]

# Create a MultiLabelBinarizer and fit it on all_label_combinations
mlb = MultiLabelBinarizer(classes=np.arange(len(all_label_combinations)))
mlb.fit(all_label_combinations)

# Function to binarize a label combination
def custom_binarize(label_combination):
    binarized = np.zeros(len(all_label_combinations))
    for idx, label_set in enumerate(all_label_combinations):
        if set(label_set) == set(label_combination):
            binarized[idx] = 1
    return binarized

# Binarize y_train and y_val using the custom binarization function
y_train_bin = np.array([custom_binarize(labels) for labels in y_train])
y_val_bin = np.array([custom_binarize(labels) for labels in y_val])

# Print the binarized arrays
print("Binarized y_train:")
print(y_train_bin)
print("Binarized y_val:")
print(y_val_bin)
