# script.py
import sys
import numpy as np
import pandas as pd
from itertools import chain, combinations
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from KNNClassifier import KNNClassifier
if len(sys.argv) < 2:
    print("Error: Input file path not provided.")
    sys.exit(1)

input_file = sys.argv[1]
data = np.load(input_file, allow_pickle=True)

vit_knn = KNNClassifier(k=3, distance_metric='euclidean',encoder_type='VIT')
vit_knn.fit(data=data)
vit_knn.predict()