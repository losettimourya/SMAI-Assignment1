import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
class KNNClassifier:
    def __init__(self, k=5, distance_metric='manhattan', encoder_type=None):
        self.k = k
        self.distance_metric = distance_metric
        self.encoder_type = encoder_type
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    def train_val_split(self,X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        split_index = int(len(X) * (1 - test_size))
        train_indices, val_indices = indices[:split_index], indices[split_index:]
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        x_train = [item[0][0] for item in X_train]
        x_val = [item[0][0] for item in X_val]
            
        return x_train, x_val, y_train, y_val

    def unshuffled_train_val_split(self,X,y,test_size=0.2):
        total_samples = len(X)
        split_index = int(total_samples * (1 - test_size))

        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        x_train = [item[0][0] for item in X_train]
        x_val = [item[0][0] for item in X_test]
        return x_train, x_val, y_train, y_test
    def fit(self, data):
        if self.encoder_type == 'VIT':
            X_vit = data[:, 2:3]
            y = data[:, 3] 
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_val_split(X_vit, y)
        elif self.encoder_type == 'Resnet':
            X_resnet = data[:, 1:2]
            y = data[:, 3] 
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_val_split(X_resnet, y)

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1-x2,axis=1,ord=2)
    def manhattan_distance(self, x1, x2):
        return np.linalg.norm(x1-x2,axis=1,ord=1)
    def cosine_distance(self, x1, x2) -> float:
        return 1-np.dot(x2,x1)/(np.linalg.norm(x2,axis=1)*np.linalg.norm(x1))
    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(x1, x2)
        else:
            raise ValueError("Unsupported distance metric")
    def fitpt2(self, data, X_test):
        if self.encoder_type == 'VIT':
            X_vit = data[:, 2:3]
            y = data[:, 3] 
            self.X_train = X_vit
            self.y_train = y
            self.X_test = X_test
        elif self.encoder_type == 'Resnet':
            X_resnet = data[:, 1:2]
            y = data[:, 3] 
            self.X_train = X_resnet
            self.y_train = y
            self.X_test = X_test
        
    def predict(self):
        y_pred = []
        for x in self.X_test:
            distances = self.calculate_distance(x, self.X_train)
            sorted_indices = np.argsort(distances)
            k_indices = sorted_indices[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            pred_label = unique_labels[np.argmax(counts)]
            y_pred.append(pred_label)
        print(self.encoder_type)
        print("Accuracy:" + str(accuracy_score(self.y_test,y_pred)))
        print("F1 Score:" + str(f1_score(self.y_test,y_pred,average='macro')))
        print("Precision score:" + str(precision_score(self.y_test,y_pred,average='macro',zero_division=0)))
        print("Recall score:" + str(recall_score(self.y_test,y_pred,average='macro',zero_division=0)))
def train_val_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    split_index = int(len(X) * (1 - test_size))
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    x_train = [item[0][0] for item in X_train]
    x_val = [item[0][0] for item in X_val]
        
    return x_train, x_val, y_train, y_val
data = np.load('data.npy', allow_pickle=True)
X_vit = data[:, 2:3]
y = data[:, 3] 
X_train, X_test, y_train, y_test = train_val_split(X_vit, y)
knn_classifier = KNeighborsClassifier()
start_time = time.time()
vit_knn = KNNClassifier(k=3, distance_metric='euclidean',encoder_type='VIT')
vit_knn.fit(data=data)
vit_knn.predict()
end_time = time.time()
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
end2_time = time.time()
resnet_knn = KNNClassifier(k=5, distance_metric='cosine',encoder_type='Resnet')
resnet_knn.fit(data=data)
resnet_knn.predict()
execution_time = end_time - start_time
execution_time2 = end2_time - end_time
print(f"Execution time1: {execution_time} seconds")
print(f"Execution time2: {execution_time2} seconds")

