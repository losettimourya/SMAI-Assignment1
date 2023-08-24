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
K = 5
accuracies = []
f1_macros = []
f1_micros = []
precisions = []
recalls = []
for _ in range(K):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/K, random_state=42+_)
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
    mlb = MultiLabelBinarizer(classes=np.arange(len(all_label_combinations)))
    mlb.fit(all_label_combinations)

    def custom_binarize(label_combination):
        binarized = np.zeros(len(all_label_combinations))
        for idx, label_set in enumerate(all_label_combinations):
            if set(label_set) == set(label_combination):
                binarized[idx] = 1
        return binarized

    y_train = np.array([custom_binarize(labels) for labels in label_combinations_list])
    y_val = np.array([custom_binarize(labels) for labels in label_combinations_val_list])

    clf = PowersetDecisionTreeClassifier(max_depth=20, max_features = 11,criterion='entropy')
    clf.fit(X_train, y_train)
    val_predictions = clf.predict(X_val)

    accuracy = accuracy_score(y_val, val_predictions)
    micro_f1 = f1_score(y_val, val_predictions, average='micro',zero_division=0)
    macro_f1 = f1_score(y_val, val_predictions, average='macro',zero_division=0)
    conf_matrix = confusion_matrix(y_val.argmax(axis=1), val_predictions.argmax(axis=1))
    precision = precision_score(y_val, val_predictions, average='micro')
    recall = recall_score(y_val, val_predictions, average='micro')
    accuracies.append(accuracy)
    f1_macros.append(macro_f1)
    f1_micros.append(micro_f1)
    precisions.append(precision)
    recalls.append(recall)
print("AVERAGES")
print("Accuracy: ",end="")
print(np.mean(accuracies))
print("Macro F1 Score: ",end="")
print(np.mean(f1_macros))
print("Micro F1 Score: ",end="")
print(np.mean(f1_micros))
print("Precision: ",end="")
print(np.mean(precisions))
print("Recall: ",end="")
print(np.mean(recalls))