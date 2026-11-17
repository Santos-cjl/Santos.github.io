import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#seaborn找配色去seaborn palette网站找
data=pd.read_csv('../data/adult.csv')

#连续型图
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(12,8))
plt.grid('--',alpha=0.5)
plt.hist(data['age'],bins=30,color='skyblue',alpha=0.7,edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Histogram')
plt.show()

#离散型变量数据图
plt.figure(figsize=(12,8))
sns.countplot(data=data,x='education',hue='income',palette='Spectral')
plt.xticks(rotation=45)
plt.tight_layout()    #整体优化
plt.show()

#连续型变量数据图
plt.figure(figsize=(12,8))
sns.kdeplot(data=data,x='age',hue='income',palette='Set2',fill=True,alpha=0.5)
plt.tight_layout()
plt.show()

#箱线图
plt.figure(figsize=(12,8))
sns.boxplot(data=data,x='income',y='age',palette='Set2')
plt.show()

label_col = 'income'
def plot_countplot(data, x_name = None, hue_name = None):
    plt.figure(figsize=(12, 8))
    sns.countplot(data=data, x=x_name, hue=hue_name, palette='Set2')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

dis_cols = data.select_dtypes(include = 'object').columns.tolist()
for col in dis_cols:
    plot_countplot(data, x_name=col, hue_name=label_col)

#相关性分析(画热力图）
corr=data.corr()   #只能分析数值型数据的相关性
plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f',square=True,cbar_kws={'label':'Correlation'})
plt.title('Correlation Heatmap')
plt.show()