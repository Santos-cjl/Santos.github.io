import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier      #导入K临近分类器（适用KNN算法）
from sklearn.tree import DecisionTreeClassifier     #导入决策树的库
from sklearn.metrics import classification_report, roc_auc_score
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score



warnings.filterwarnings('ignore')

# Load the dataset
data=pd.read_csv('../../data/adult.csv')

#数据预处理
dis_cols = data.select_dtypes(include=["object"]).columns.tolist()
for col in dis_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

label_col = 'income'
# train-test split
x = data.drop(label_col, axis=1)
y = data[label_col]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# clf = LogisticRegression()  采取的是逻辑回归
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_proba = clf.predict_proba(x_test)
print(classification_report(y_test, y_pred))
print('auc:', roc_auc_score(y_test, y_proba[:, 1]))

#对比模型之间的性能（通过auc的大小粗略估计）
def model_pred(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)[:, 1]
    print(clf.__class__.__name__)
    print(classification_report(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_proba))

model_pred(LogisticRegression(), x_train, y_train, x_test, y_test)
model_pred(KNeighborsClassifier(), x_train, y_train, x_test, y_test)
model_pred(DecisionTreeClassifier(random_state=0), x_train, y_train, x_test, y_test)
model_pred(RandomForestClassifier(n_estimators=100, random_state=0), x_train, y_train, x_test, y_test)
model_pred(GradientBoostingClassifier(n_estimators=100, random_state=0), x_train, y_train, x_test, y_test)
model_pred(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0), x_train, y_train, x_test, y_test)