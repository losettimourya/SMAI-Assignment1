import numpy as np
import pandas as pd
from itertools import chain, combinations
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

class PowersetDecisionTreeClassifier:
    def __init__(self, max_depth=5, max_features='auto', criterion='gini'):
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.classifier = DecisionTreeClassifier(max_depth=max_depth,max_features=max_features,criterion=criterion,random_state=42)
    def fit(self, X, y):
        self.classifier.fit(X, y)
    def predict(self, X):
        return self.classifier.predict(X)
data = pd.read_csv('advertisement.csv')  # Replace with your CSV file path
# print("Original DataFrame:")
# print(data.head())
data.fillna(method='ffill', inplace=True) 
categorical_cols = ['gender', 'education', 'married', 'city', 'occupation', 'most bought item']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
# print("Encoded DataFrame:")
# print(data_encoded.head())
X = data_encoded.drop('labels', axis=1)  # Features
y = data_encoded['labels']  # Target variable
# print(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# print(y_train)
def powerset(iterable):
    s = list(iterable)
    unique_combos = set()
    for r in range(len(s),len(s) + 1):
        for combo in combinations(sorted(s), r):
            return tuple((combo))
            unique_combos.add(tuple(combo))
    return list(unique_combos)
label_combinations = y_train.apply(lambda x: tuple(x.split(' '))).apply(powerset)
label_combinations_val = y_val.apply(lambda x: tuple(x.split(' '))).apply(powerset)
# print(label_combinations)

all_label_combinations = list(set(label_combinations.sum() + label_combinations_val.sum()))
unique_labels = y.str.split(' ').explode().unique()  
all_label_combinations = list(chain.from_iterable(combinations(unique_labels, r) for r in range(len(unique_labels) + 1)))
label_combinations_list = label_combinations.tolist()
label_combinations_val_list = label_combinations_val.tolist()
#print(label_combinations_list)
# print(label_combinations_val_list)
# all_label_combinations.sort()
all_label_combinations = sorted(all_label_combinations, key=lambda x: tuple(sorted(x)))
print((all_label_combinations))
mlb = MultiLabelBinarizer(classes=np.arange(len(all_label_combinations)))
mlb.fit(all_label_combinations)

def custom_binarize(label_combination):
    binarized = np.zeros(len(all_label_combinations))
    for idx, label_set in enumerate(all_label_combinations):
        if set(label_set) == set(label_combination):
            binarized[idx] = 1
    return binarized

# Binarize y_train and y_val using the custom binarization function
# label_combinations_list.sort()
# label_combinations_val_list.sort()
# print(label_combinations_list)
y_train = np.array([custom_binarize(labels) for labels in label_combinations_list])
y_val = np.array([custom_binarize(labels) for labels in label_combinations_val_list])
# print(y_val)
# y_train = mlb.fit_transform(label_combinations_list)
# y_train_df = pd.DataFrame(y_train, columns=mlb.classes_)
# y_train = y_train_df.loc[:, ~y_train_df.columns.duplicated()]
# y_val = mlb.fit_transform(label_combinations_val_list)
# y_val_df = pd.DataFrame(y_val, columns=mlb.classes_)
# # # Remove duplicate columns, if any
# y_val = y_val_df.loc[:, ~y_val_df.columns.duplicated()]
# # y_val = mlb.fit_transform(y_val.apply(lambda x: tuple(x.split(' '))))
#print(y_train)
# print(y_val)
print(np.sum(y_val[1]))
clf = PowersetDecisionTreeClassifier(max_depth=30, max_features = 11,criterion='gini')
clf.fit(X_train, y_train)
# print(X_train)
for i in range(0,len(y_train[0])):
    if(y_train[0][i] == 1):
        print(i)
        break
val_predictions = clf.predict(X_val)
# print(val_predictions)
# y_val = y_val.to_numpy()
print(np.sum(y_train))
print(np.sum(y_val))
# print(val_predictions)
print(np.sum(val_predictions))
accuracy = accuracy_score(y_val, val_predictions)
micro_f1 = f1_score(y_val, val_predictions, average='micro',zero_division=0)
macro_f1 = f1_score(y_val, val_predictions, average='macro',zero_division=0)
conf_matrix = confusion_matrix(y_val.argmax(axis=1), val_predictions.argmax(axis=1))
precision = precision_score(y_val, val_predictions, average='micro')
recall = recall_score(y_val, val_predictions, average='micro')

# print(f'Accuracy: {accuracy:.5f}')
# print(f'F1 (Micro): {micro_f1:.5f}')
# print(f'F1 (Macro): {macro_f1:.5f}')
# print('Confusion Matrix:')
# print(conf_matrix)
# print(f'Precision (Micro): {precision:.2f}')
# print(f'Recall (Micro): {recall:.2f}')
# val_predictions = mlb.inverse_transform(val_predictions)
# y_val = mlb.inverse_transform(y_val)

# Initialize an empty dictionary to store confusion matrices for each label
# confusion_matrices = {}

# # Compute confusion matrix for each label
# for label_idx, label in enumerate(mlb.classes_):
#     y_val_label = [1 if label in labels else 0 for labels in y_val]
#     val_predictions_label = [1 if label in labels else 0 for labels in val_predictions]
#     cm = confusion_matrix(y_val_label, val_predictions_label)
#     confusion_matrices[label] = cm

# # Print confusion matrices for each label
# for label, cm in confusion_matrices.items():
#     print(f'Confusion Matrix for Label "{label}":')
#     print(cm)