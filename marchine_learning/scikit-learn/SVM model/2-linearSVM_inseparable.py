import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D  # 绘图库的拓展，三维空间
import numpy as np  # 科学计算，多维数组和矩阵


# 导入数据集
X, y = make_blobs(centers=4, random_state=8)
y = y % 2

plt.rcParams['font.size'] = 15
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 数据集可视化
plt.figure(1)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1"], loc=4)
plt.title("Raw data")
plt.show()

# 模型优化学习
linear_svm = LinearSVC().fit(X, y)
plt.figure(2)
mglearn.plots.plot_2d_separator(linear_svm, X)  # 画决策边界
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)  # 画数据点
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1"], loc=4)
plt.show()

# 添加第二个特征的平方作为一个新特征
X_new = np.hstack([X, X[:, 1:] ** 2])
# 3D可视化
fig3 = plt.figure(3)
ax = Axes3D(fig3, elev=-152, azim=-26)  # 创建3D轴对象
fig3.add_axes(ax)
mask = y == 0  # 创建掩码，用来区分两类数据
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', s=60)  # y==0
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', s=60)  # y==1
ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
ax.set_zlabel("Feature 2")
plt.show()

# 模型再优化学习
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_  # 获取模型的权重系数和截距
fig4 = plt.figure(4)
ax = Axes3D(fig4, elev=-152, azim=-26)  # 创建3D轴对象
fig4.add_axes(ax)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)  # 创建网格点
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)  # 生成网格数据
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]  # 计算决策边界的高度
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)  # 画决策边界
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)  # y==0
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)  # y==1
ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
ax.set_zlabel("Feature 1 ** 2")
plt.show()
