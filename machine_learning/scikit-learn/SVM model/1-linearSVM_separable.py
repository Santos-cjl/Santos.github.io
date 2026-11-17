# 导入库
import mglearn  # 数据集的加载、可视化的工具和辅助函数
import matplotlib.pyplot as plt  # 绘图 2D
from sklearn.svm import LinearSVC,SVC,SVR  # 支持向量机分类、回归

# 导入数据集
X, y = mglearn.datasets.make_forge()

# 数据集可视化
plt.figure(1)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)  # 画数据散点图
plt.legend()
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.title("Raw data")
plt.show()
print("X.shape:{}".format(X.shape))

# 模型优化学习
model = LinearSVC()
clf = model.fit(X, y)  # 训练
plt.figure(2)
mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, alpha=0.7)  # 画分类的决策边界
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.title("{}".format(clf.__class__.__name__))
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1"], loc=4)
plt.show()

# 线性SVM在不同C参数下分类结果的可视化
plt.figure(3)
mglearn.plots.plot_linear_svc_regularization()
plt.show()
