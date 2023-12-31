import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

class MultiOutputDecisionTreeClassifier:
    def __init__(self, max_depth=5, max_features='auto', criterion='gini'):
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.classifier = DecisionTreeClassifier(max_depth=max_depth,max_features=max_features,criterion=criterion,random_state=42)
    def fit(self, X, y):
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y.str.split(' '))
        self.classifier.fit(X, y)
    def predict(self, X):
        return self.classifier.predict(X)
data = pd.read_csv('advertisement.csv')  # Replace with your CSV file path
data.fillna(method='ffill', inplace=True) 
categorical_cols = ['gender', 'education', 'married', 'city', 'occupation', 'most bought item']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
X = data_encoded.drop('labels', axis=1)  # Features
y = data_encoded['labels']  # Target variable
K = 5
validation_indices = []
n = len(X)
for i in range(K):
    start = (n // K) * i
    end = (n // K) * (i + 1)
    val_indices = list(range(start, end))
    train_indices = list(set(range(n)) - set(val_indices))
    validation_indices.append((train_indices, val_indices))
accuracies = []
f1_macros = []
f1_micros = []
precisions = []
recalls = []
for i,(train_indices,val_indices) in enumerate(validation_indices):
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/K,random_state=42+_)
    # train_indices = np.setdiff1d(np.arange(n), val_indices)
    # print(train_indices)
    # print(val_indices)
    # print(X.iloc[train_indices])
    # print(X.iloc[val_indices])
    X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
    # y_train, y_val = y[train_indices], y[val_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
    mlb = MultiLabelBinarizer()
    y_val = mlb.fit_transform(y_val.str.split(' '))
    clf = MultiOutputDecisionTreeClassifier(max_depth=30, max_features = 1100,criterion='entropy')
    clf.fit(X_train, y_train)
    val_predictions = clf.predict(X_val)
    # print(val_predictions)
    accuracy = accuracy_score(y_val, val_predictions)
    micro_f1 = f1_score(y_val, val_predictions, average='micro',zero_division=0)
    macro_f1 = f1_score(y_val, val_predictions, average='macro',zero_division=0)
    conf_matrix = confusion_matrix(y_val.argmax(axis=1), val_predictions.argmax(axis=1))
    precision = precision_score(y_val, val_predictions, average='micro',zero_division=0)
    recall = recall_score(y_val, val_predictions, average='micro',zero_division=0)
    # print(accuracy)
    accuracies.append(accuracy)
    f1_macros.append(macro_f1)
    f1_micros.append(micro_f1)
    precisions.append(precision)
    recalls.append(recall)
    # print(f'Accuracy: {accuracy:.2f}')
    # print(f'F1 (Micro): {micro_f1:.2f}')
    # print(f'F1 (Macro): {macro_f1:.2f}')
    # print('Confusion Matrix:')
    # print(conf_matrix)
    # print(f'Precision (Micro): {precision:.2f}')
    # print(f'Recall (Micro): {recall:.2f}')
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
