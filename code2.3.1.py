import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
        # x1 = np.array(x1)
        # x2 = np.array(x2)
        # print(len(x1))
        # print(len(x2))
        # x22 = []
        # for i in range(0,len(x2)):
        #     x22.append(x2[i][0][0])
        # x22 = np.vectorize(lambda x: x[0][0])(x2)
        #print(len(x22[0]))
        return np.linalg.norm(x1-x2,axis=1,ord=2)
        # return np.sqrt(np.sum((x1 - x2) ** 2),axis=1)

    def manhattan_distance(self, x1, x2):
        return np.linalg.norm(x1-x2,axis=1,ord=1)
    def cosine_distance(self, x1, x2) -> float:
        #dot_product = np.dot(x1, x2)
        #print(dot_product)
        # norm_x1 = np.linalg.norm(x1)
        # norm_x2 = np.linalg.norm(x2,axis=1)
        # #print(len(norm_x2))
        # dot_product = np.dot(x2,x1)
        # #print(len(dot_product))
        # similarities = dot_product/(norm_x1 * norm_x2)
        # print(len(similarities))
        return 1-np.dot(x2,x1)/(np.linalg.norm(x2,axis=1)*np.linalg.norm(x1))
        #return np.sum(1-similarities)

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
        print(self.encoder_type)
        print("Accuracy:" + str(accuracy_score(self.y_test,y_pred)))
        print("F1 Score:" + str(f1_score(self.y_test,y_pred,average='macro')))
        print("Precision score:" + str(precision_score(self.y_test,y_pred,average='macro',zero_division=0)))
        print("Recall score:" + str(recall_score(self.y_test,y_pred,average='macro',zero_division=0)))
        #return np.array(y_pred)

# Load the dataset from data.npy
data = np.load('data.npy', allow_pickle=True)
# X_resnet = data[:, 1:2] 
# X_vit = data[:, 2:3]
# y = data[:, 3] 
    

# X_resnet_train, X_resnet_test, y_resnet_train, y_resnet_test = train_val_split(X_resnet, y)
# X_vit_train, X_vit_test, y_vit_train, y_vit_test = train_val_split(X_vit, y)
vit_knn = KNNClassifier(k=3, distance_metric='euclidean',encoder_type='VIT')
vit_knn.fit(data=data)
vit_knn.predict()
resnet_knn = KNNClassifier(k=5, distance_metric='cosine',encoder_type='Resnet')
resnet_knn.fit(data=data)
resnet_knn.predict()
