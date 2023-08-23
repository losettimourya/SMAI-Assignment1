import pandas as pd
from itertools import chain, combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Load your data (replace 'advertisement.csv' with your CSV file path)
data = pd.read_csv('advertisement.csv')
print("Original DataFrame:")
print(data.head())

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Define categorical columns and one-hot encode them
categorical_cols = ['gender', 'education', 'married', 'city', 'occupation', 'most bought item']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print("Encoded DataFrame:")
print(data_encoded.head())

# Split data into features (X) and targets (y)
X = data_encoded.drop('labels', axis=1)  # Features
y = data_encoded['labels']  # Target variable

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a MultiLabelBinarizer to transform label combinations into a binary matrix
mlb = MultiLabelBinarizer()

# Get unique labels from X_train and X_val
unique_labels_train = set(chain.from_iterable(X_train.columns.str.split()))
unique_labels_val = set(chain.from_iterable(X_val.columns.str.split()))

# Combine unique labels from both sets to get all possible labels
unique_labels = unique_labels_train | unique_labels_val

# Generate all possible label combinations (power set)
all_label_combinations = list(chain.from_iterable(combinations(unique_labels, r) for r in range(len(unique_labels) + 1)))

# Convert label combinations to a list of lists
# label_combinations_list = [list(comb) for comb in all_label_combinations]

# # Transform label combinations into binary matrices for y_train and y_val
# y_train_bin = mlb.fit_transform([list(set(x.split()) & set(unique_labels)) for x in y_train])
# y_val_bin = mlb.transform([list(set(x.split()) & set(unique_labels)) for x in y_val])

# # Create dataframes from binary matrices
# y_train_df = pd.DataFrame(y_train_bin, columns=mlb.classes_)
# y_val_df = pd.DataFrame(y_val_bin, columns=mlb.classes_)

# # Remove duplicate columns, if any
# y_train_df = y_train_df.loc[:, ~y_train_df.columns.duplicated()]
# y_val_df = y_val_df.loc[:, ~y_val_df.columns.duplicated()]

# # Create and train the DecisionTreeClassifier for the powerset formulation using common columns
# clf = DecisionTreeClassifier(max_depth=5, random_state=42)
# clf.fit(X_train, y_train_df)

# # Make predictions on the validation set
# val_predictions = clf.predict(X_val)
