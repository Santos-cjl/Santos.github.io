import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the dataset
data = pd.read_csv("../../data/UCI_Credit_Card.csv")
label_col = data.columns[-1]

if 'ID' in data.columns:
    data.drop('ID', axis=1, inplace=True)

#针对数据集中的object类型的数据转换成数值型
dis_cols = data.select_dtypes(include=['object']).columns.tolist()
encoder_method = "label_encoding"  # Options: "label_encoding" or "onehot"
if encoder_method == "label_encoding":
    for col in dis_cols:
        data[col] = LabelEncoder().fit_transform(data[col])
elif encoder_method == "onehot":    #独热编码
    for col in dis_cols:
        if col == label_col:
            data[col] = LabelEncoder().fit_transform(data[col])
        else:
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummies], axis=1)
            data.drop(col, axis=1, inplace=True)

x = data.drop(label_col, axis=1)
y = data[label_col]

corr = data.corr()
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=[12,12])
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, annot_kws={"size":8})
plt.title('Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0, stratify=y)
models_name  = ['LR', 'DT', 'RF', 'GBDT', 'XGBoost']
models = [
    LogisticRegression(max_iter=1000, random_state=0),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(n_estimators=150, random_state=0),
    GradientBoostingClassifier(n_estimators=150, random_state=0),
    XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric='logloss', random_state=0)
]

results = pd.DataFrame(columns=models_name, index=['Accuracy', 'AUC', 'F1-Score', 'Precision', 'Recall', 'time'])

for model, name in zip(models, models_name):
    begin_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    results.loc['AUC', name] = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    results.loc['Accuracy', name] = report['accuracy']
    results.loc['F1-Score', name] = report['1']['f1-score']
    results.loc['Precision', name] = report['1']['precision']
    results.loc['Recall', name] = report['1']['recall']
    results.loc['time', name] = end_time - begin_time

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Times New Roman'
results.drop('time').plot(kind='bar', color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], edgecolor='black', linewidth=1)
plt.ylim([results.drop('time').min().min()-0.01,results.drop('time').max().max()+0.01])
plt.grid('--', alpha=0.5)
plt.title('Model Comparison on Adult Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()