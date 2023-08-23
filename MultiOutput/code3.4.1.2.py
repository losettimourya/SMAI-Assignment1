import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

class MultiOutputDecisionTreeClassifier:
    def __init__(self, max_depth, max_features, criterion):
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.classifier = DecisionTreeClassifier(max_depth=max_depth,max_features=max_features,criterion=criterion,random_state=42)
    def fit(self, X, y):
        self.classifier.fit(X, y)
    def predict(self, X):
        return self.classifier.predict(X)
data = pd.read_csv('advertisement.csv') 
data.fillna(method='ffill', inplace=True) 
categorical_cols = ['gender', 'education', 'married', 'city', 'occupation', 'most bought item']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
X = data_encoded.drop('labels', axis=1) 
y = data_encoded['labels'] 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
# test_size = 0.2

# # Calculate the number of data points to reserve for testing
# num_test_samples = int(len(X) * test_size)

# # Create the training and testing sets without shuffling
# X_train, X_val = X[:num_test_samples], X[num_test_samples:]
# y_train, y_val = y[:num_test_samples], y[num_test_samples:]
mlb = MultiLabelBinarizer()
# y_train = y_train.str.get_dummies(sep=' ')
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)
ans = []
for i in [3,5,10,20,30]:
    for j in [3,5,7,9,11]:
        for k in ['gini','entropy']:
            clf = MultiOutputDecisionTreeClassifier(max_depth=i, max_features = j,criterion=k)
            clf.fit(X_train, y_train)
            val_predictions = clf.predict(X_val)
            # print(y_val)
            macro_f1 = f1_score(y_val, val_predictions, average='macro')
            ans.append([macro_f1,i,j,k])
ans.sort(reverse=True)
for i in range(0,3):
    print("F1 Score: " + str(ans[i][0]) + " Max Depth: " + str(ans[i][1]) + " Max Features: " + str(ans[i][2]) + " Criterion: " + str(ans[i][3]))
