import pandas as pd
#一、创建一个dataframe
a1=pd.DataFrame({'name':['Alice','Bob','Charlie'],
                 'age':[25,30,35],
                 'city':['London','Paris','New York']},
                index=['a','b','c'])

data=pd.read_csv('../data/adult.csv')

#二、DataFrame description
print(data.describe())  #获取数值型数据的统计信息
print(data.info())     #主要收集数据集的基本信息，每列的数据类型，和非空值计数
print(data.head())     #前5行
print(data.tail())     #最后5行
print(data.shape)      #数据集大小
print(data.columns)    #获取列名
print(data.dtypes)     #获取数据类型
print(data.index)      #获取每列的数据类型

#三、DataFrame indexing and selection
print(data['age'])     #打印列
print(data[['age','education']])
print(data.iloc[:,0])  #打印行
#ilock是按照位置读取，lock是按照名字去读取
print(data.iat[0,1])   #选择第一行第二列的值

#四、DataFrame Sort
data.sort_values('age',ascending=False)  #根据值排列
#ascending=False是降序,ascending=Ture是升序。

#五、DataFrame row and column operations
data.sample(n=100)  # 随机抽取100行数据
data.sample(frac=0.1)  # 随机抽取10%的数据
data.nlargest(5, 'age')  # 获取'age'列最大的5行数据
data.nsmallest(5, 'age')  # 获取'age'列最小的5行数据
data.query('age > 30')  # 使用查询语句筛选数据
data.query('age < 30 and fnlwgt > 100000')  # 使用查询语句筛选数据
data1 = data[data['age'] > 30]  # 使用布尔索引筛选数据
data2 = data[data['education.num'] > 10]  # 使用布尔索引筛选数据
data.query('index in@data1.index and index in@data2.index')

# 六、DataFrame data analysis
print(f"age 列的平均值: {data['age'].mean()}")  # 计算'age'列的平均值
print(f"age 列的最大值: {data['age'].max()}")  # 计算'age'列的最大值
print(f"age 列的最小值: {data['age'].min()}")  # 计算'age'列的最小值
print(f"age 列的标准差: {data['age'].std()}")  # 计算'age'列的标准差
print(f"age 列的方差: {data['age'].var()}")  # 计算'age'列的方差
print(f"age 列的中位数: {data['age'].median()}")  # 计算'age'列的中位数
print(f"age 列的唯一值数量: {data['age'].nunique()}")  # 计算'age'列的唯一值数量
data['age'].value_counts()  # 计算'age'列的值计数
data['age'].unique()  # 获取'age'列的唯一值

# 七、DataFrame groupby
grouped_data = data.groupby('education')['age'].mean()  # 按照'education'列分组并计算'age'列的平均值
data.groupby('education').agg({'age': ['mean', 'max', 'min'], 'education.num': 'sum'})  # 按照'education'列分组并计算'age'列的平均值、最大值、最小值和'education.num'列的总和

# 八、DataFrame merge and join
a2 = pd.DataFrame([[1,2,3],[4,5,6]], columns=['A','B','C'])
a3 = pd.DataFrame([[7,8,9],[10,11,12]], columns=['A','B','C'])
a4 = pd.concat([a2, a3])     #行拼接
a5 = pd.concat([a2, a3],axis=1)    #列拼接

# 九、DataFrame visualization
import matplotlib.pyplot as plt
data['age'].hist(bins=20)  # 绘制'age'列的直方图
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()