import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_openml

#load the dataset
df=fetch_openml('house_prices',as_frame=True)
x=df['data']
x.drop('Id',axis=1,inplace=True)   #数据集中的Id列均是同质类的，可以删除
y=df['target']
data=pd.concat([x,y],axis=1)  #删除后可以进行一个拼接

#Check for missing values
x.isnull().sum()
missing_ratio=x.isnull().sum()/len(x)    #缺失比
missing_x=missing_ratio[missing_ratio > 0] #筛选出有缺失值的x数据
del_cols=missing_x[missing_ratio > 0.8].index.tolist()   #筛选出缺失值大于80%的x数据
data.drop(del_cols,axis=1,inplace=True)    #采取删除的方法直接删除数据集

#采用遍历后众数填充的方法进行数据的预处理
missing_cols=list(set(missing_x.index)-set(del_cols))
for col in missing_cols:
    data[col].fillna(data[col].mode()[0],inplace=True)

#检验是否还有缺失值
data.isnull().sum().sort_values(ascending=False)
data.isnull().sum().sum()  #看到最后总和为0即数据处理成功

#visualize the discrete features VS the target
dis_cols=data.select_dtypes(include=['object']).columns.tolist()
plt.rcParams['font.size'] = 15
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize=[8,8])
sns.kdeplot(data=data,x='SalePrice',hue='Street',fill=True)
plt.show()

#visualise the continuous features VS the target
con_cols=data.select_dtypes(exclude=['object']).columns.tolist()
con_cols.remove('SalePrice')
for col in con_cols:
    plt.figure(figsize=[8,8])
    sns.jointplot(data=data,x=col,y='SalePrice',kind='reg')
    plt.show()    #批量循环作图
