import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
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
        # return np.sqrt(np.sum((x1 - x2) ** 2),axis=1)
        return np.linalg.norm(x1-x2,axis=1,ord=2)

    def manhattan_distance(self, x1, x2):
        return np.linalg.norm(x1-x2,axis=1,ord=1)

    def cosine_distance(self, x1, x2):
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
        #print(len(X_test[0][0]))
        #print(X_test)
        for x in self.X_test:
            #print(x)
            #distances = [self.calculate_distance(x[0][0], x_train[0][0]) for x_train in self.X_train]
            #self.X_train = self.X_train[0]
            # print(len(self.X_train))
            # print(len(x[0][0]))
            distances = self.calculate_distance(x, self.X_train)
            # distances = np.array([self.calculate_distance(x[0][0], x_train[0][0]) for x_train in self.X_train])
            sorted_indices = np.argsort(distances)
            k_indices = sorted_indices[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            pred_label = unique_labels[np.argmax(counts)]
            y_pred.append(pred_label)
        return accuracy_score(self.y_test,y_pred)
        #return np.array(y_pred)

# Load the dataset from data.npy
data = np.load('data.npy', allow_pickle=True)
# X_resnet = data[:, 1:2] 
# X_vit = data[:, 2:3]
# y = data[:, 3] 
    

# X_resnet_train, X_resnet_test, y_resnet_train, y_resnet_test = train_val_split(X_resnet, y)
# X_vit_train, X_vit_test, y_vit_train, y_vit_test = train_val_split(X_vit, y)
k_value = []
accuracy = []
for i in range(1,10,2):
    knn = KNNClassifier(k=i,distance_metric='euclidean',encoder_type='VIT')
    knn.fit(data=data)
    ans = knn.predict()
    k_value.append(i)
    accuracy.append(ans)

print(k_value)
print(accuracy)
plt.plot(k_value, accuracy, marker='o', linestyle='-')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('K vs Accuracy graph')
plt.show()
# def calculate_accuracy(y_true, y_pred):
#     # Implement accuracy calculation
#     return accuracy_score(y_true,y_pred)

# def calculate_f1_score(y_true, y_pred):
#     # Implement F1 score calculation
#     return f1_score(y_true,y_pred,average='macro')

# def calculate_precision(y_true, y_pred,average='macro'):
#     # Implement precision calculation
#     return precision_score(y_true,y_pred,average='macro')

# def calculate_recall(y_true, y_pred):
#     # Implement recall calculation
#     return recall_score(y_true,y_pred,average='macro')

# accuracy_vit = accuracy_score(y_vit_test, y_pred_vit,average='macro')
# f1_score_vit = f1_score(y_vit_test, y_pred_vit,average='macro')
# precision_vit = precision_score(y_vit_test, y_pred_vit,average='macro')
# recall_vit = recall_score(y_vit_test, y_pred_vit,average='macro')

# accuracy_resnet = accuracy_score(y_resnet_test, y_pred_resnet,average='macro')
# f1_score_resnet = f1_score(y_resnet_test, y_pred_resnet,average='macro')
# precision_resnet = precision_score(y_resnet_test, y_pred_resnet,average='macro')
# recall_resnet = recall_score(y_resnet_test, y_pred_resnet,average='macro')
# # print(y_pred_resnet)
# # print(y_resnet_test)
# print("Metrics for VIT-based KNN Classifier:")
# print(f"Accuracy: {accuracy_vit}")
# print(f"F1 Score: {f1_score_vit}")
# print(f"Precision: {precision_vit}")
# print(f"Recall: {recall_vit}")

# print("\nMetrics for ResNet-based KNN Classifier:")
# print(f"Accuracy: {accuracy_resnet}")
# print(f"F1 Score: {f1_score_resnet}")
# print(f"Precision: {precision_resnet}")
# print(f"Recall: {recall_resnet}")
