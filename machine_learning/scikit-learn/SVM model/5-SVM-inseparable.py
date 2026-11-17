import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split  # 模块提供用于模型选择和评估的工具

# 导入数据集
cancer = load_breast_cancer()

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# 可视化训练集数据的范围
plt.figure(1)
plt.plot(X_train.min(axis=0), 'o', label="min")  # 画每个特征的最小值
plt.plot(X_train.max(axis=0), '^', label="max")  # 画每个特征的最大值
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")
plt.show()

# SVM
svc = SVC()

# 模型优化学习
svc.fit(X_train, y_train)
print("Accuracy on training set: {:,.2f}".format(svc.score(X_train, y_train)))  # 输出训练结果
print("Accuracy on test set: {:,.2f}".format(svc.score(X_test, y_test)))  # 输出训练结果

# 数据预处理（最大最小归一化）
min_on_training = X_train.min(axis=0)  # 计算训练集每个特征的最小值
range_on_training = (X_train - min_on_training).max(axis=0)  # 计算特征的范围
X_train_scaled = (X_train - min_on_training) / range_on_training  # 归一化
print("Minimum for each feature \n{}".format(X_train_scaled.min(axis=0)))
print("Maximum for each feature \n{}".format(X_train_scaled.max(axis=0)))
X_test_scaled = (X_test - min_on_training) / range_on_training  # 测试集归一化

# 模型再优化学习
svc = SVC()

# 模型优化学习
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:,.2f}".format(svc.score(X_train_scaled, y_train)))  # 输出训练结果
print("Accuracy on test set: {:,.2f}".format(svc.score(X_test_scaled, y_test)))  # 输出训练结果

# 调参数C
svc = SVC(C=1)

# 模型优化学习
svc.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:,.2f}".format(svc.score(X_train_scaled, y_train)))  # 输出训练结果
print("Accuracy on test set: {:,.2f}".format(svc.score(X_test_scaled, y_test)))  # 输出训练结果
