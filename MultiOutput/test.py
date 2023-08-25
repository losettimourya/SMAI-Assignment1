import numpy as np
from sklearn.model_selection import train_test_split

# Example data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0])  # Binary labels

# Number of splits for K-Fold cross-validation
K = 5

# Create an array to track used indices
used_indices = np.zeros(len(X), dtype=bool)

# Perform K-Fold cross-validation using train_test_split
for i in range(K):
    # Find an unused index for validation
    # val_index = np.argmax(~used_indices)
    
    # # Set the chosen index as used
    # used_indices[val_index] = True
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/K, random_state=i)
    
    print(f"Fold {i + 1} - Training: {X_train}, Validation: {X_val}")
