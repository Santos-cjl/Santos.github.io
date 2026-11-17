import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv("../../data/adult.csv")
# Data preprocessing
dis_cols = data.select_dtypes(include=["object"]).columns.tolist()
for col in dis_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

label_col = data.columns[-1]
# train-test split
x = data.drop(label_col, axis=1)
y = data[label_col]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define SVM classifier
svm_classifier = SVC(C = 1, kernel='rbf', gamma='scale', probability=True)  # Using RBF kernel
# Train the classifier
svm_classifier.fit(x_train, y_train)
# Make predictions
y_pred = svm_classifier.predict(x_test)
# Evaluate the classifier
print(classification_report(y_test, y_pred))