import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression   #导入线性回归和逻辑回归的库
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

#开始进行数据预处理
df = fetch_openml(name="house_prices", as_frame=True)
x = df['data']
y = df['target']
x.drop('Id', axis=1, inplace=True)

cols_str = x.select_dtypes(include=["object"]).columns
for col in cols_str:
    x[col] = LabelEncoder().fit_transform(x[col])

#缺失值处理
missing_ratio = x.isnull().sum()/len(x)
missing_x = missing_ratio[missing_ratio>0]
del_cols = missing_x[missing_x > 0.8].index.tolist()
x.drop(del_cols, axis=1, inplace=True)
missing_cols = list(set(missing_x.index) - set(del_cols))
for col in missing_cols:
    x[col].fillna(x[col].mode()[0], inplace=True)

#开始做回归预测
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf=LinearRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))   #检查回归效果，通过RMSE值来看，越大说明距离越远，根据不同模型判定效果。

plt.figure(figsize=(12, 8))
plt.plot(y_test.values, label='True Values', color='blue')   #画出实际的图
plt.plot(y_pred, label='Predicted Values', color='orange')   #画出回归预测的图
plt.title('House Price Prediction')
plt.legend()
plt.show()