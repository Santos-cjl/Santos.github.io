import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsembleClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score,accuracy_score,
                             precision_score,recall_score)
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv("data/adult.csv")
label_col = data.columns[-1]

# preprocess the dataset
dis_cols = data.select_dtypes(include='object').columns.tolist()
for col in dis_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

x = data.drop(label_col, axis=1)
y = data[label_col]
imbalance_methods = ['None', 'Over-sampling', 'Under-sampling', 'SMOTE', 'Class Weight']
models = {
    "LR": LogisticRegression(),
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

recall = pd.DataFrame(columns=models.keys(), index=imbalance_methods)
precision = pd.DataFrame(columns=models.keys(), index=imbalance_methods)
auc = pd.DataFrame(columns=models.keys(), index=imbalance_methods)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

for method in imbalance_methods:
    if method != 'Class Weight':
        if method == 'None':
            x_res, y_res = x_train, y_train
        elif method == 'Over-sampling':
            ros = RandomOverSampler(random_state=0)
            x_res, y_res = ros.fit_resample(x_train, y_train)
        elif method == 'Under-sampling':
            rus = RandomUnderSampler(random_state=0)
            x_res, y_res = rus.fit_resample(x_train, y_train)
        elif method == 'SMOTE':
            smote = SMOTE(random_state=0)
            x_res, y_res = smote.fit_resample(x_train, y_train)

        for model_name, model in models.items():
            model.fit(x_res, y_res)
            y_pred = model.predict(x_test)
            recall.loc[method, model_name] = recall_score(y_test, y_pred)
            precision.loc[method, model_name] = precision_score(y_test, y_pred)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(x_test)[:, 1]
                auc.loc[method, model_name] = roc_auc_score(y_test, y_proba)
            else:
                auc.loc[method, model_name] = roc_auc_score(y_test, model.decision_function(x_test))
    else:
        models = {
            "LR": LogisticRegression(class_weight={0:1, 1:5}),
            "DT": DecisionTreeClassifier(class_weight={0:1, 1:5}),
            "RF": RandomForestClassifier(class_weight={0:1, 1:5}),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',class_weight={0:1, 1:5})
        }
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            recall.loc[method, model_name] = recall_score(y_test, y_pred)
            precision.loc[method, model_name] = precision_score(y_test, y_pred)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(x_test)[:, 1]
                auc.loc[method, model_name] = roc_auc_score(y_test, y_proba)
            else:
                auc.loc[method, model_name] = roc_auc_score(y_test, model.decision_function(x_test))