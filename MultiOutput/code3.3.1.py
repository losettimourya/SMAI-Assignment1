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
        y = mlb.fit_transform(y)
        self.classifier.fit(X, y)
    def predict(self, X):
        return self.classifier.predict(X)
data = pd.read_csv('advertisement.csv')  # Replace with your CSV file path
print("Original DataFrame:")
print(data.head())
data.fillna(method='ffill', inplace=True) 
categorical_cols = ['gender', 'education', 'married', 'city', 'occupation', 'most bought item']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print("Encoded DataFrame:")
print(data_encoded.head())
X = data_encoded.drop('labels', axis=1)  # Features
y = data_encoded['labels']  # Target variable
print(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train)
mlb = MultiLabelBinarizer()
# y_train = y_train.str.get_dummies(sep=' ')
# y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)
# y_val = y_val.str.get_dummies(sep=' ')
# print(y_train)
print(y_val)
clf = MultiOutputDecisionTreeClassifier(max_depth=5, max_features = 'auto',criterion='gini')
clf.fit(X_train, y_train)
val_predictions = clf.predict(X_val)
print(val_predictions)
accuracy = accuracy_score(y_val, val_predictions)
micro_f1 = f1_score(y_val, val_predictions, average='micro')
macro_f1 = f1_score(y_val, val_predictions, average='macro')
conf_matrix = confusion_matrix(y_val.argmax(axis=1), val_predictions.argmax(axis=1))
precision = precision_score(y_val, val_predictions, average='micro')
recall = recall_score(y_val, val_predictions, average='micro')

print(f'Accuracy: {accuracy:.2f}')
print(f'F1 (Micro): {micro_f1:.2f}')
print(f'F1 (Macro): {macro_f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print(f'Precision (Micro): {precision:.2f}')
print(f'Recall (Micro): {recall:.2f}')
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
