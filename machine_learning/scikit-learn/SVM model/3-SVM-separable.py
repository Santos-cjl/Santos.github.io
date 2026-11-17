import mglearn
import matplotlib.pyplot as plt
from  sklearn.svm import SVC

#C为正则化参数，控制对分类错误的容忍度，C越大模型越倾向于严格分类
#gamma为RBF核的参数，控制单个样本对决策边界的影响范围。gamma越大，样本影响范围越小。

# 导入数据集
X, y = mglearn.tools.make_handcrafted_dataset()

# SVM
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)  # 高斯核

# 分类结果可视化
plt.figure(1)
mglearn.plots.plot_2d_separator(svm, X, eps=0.5)  # 决策函数绘制
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)  # 原始数据绘制
sv = svm.support_vectors_  # 获取支持向量的坐标
sv_labels = svm.dual_coef_.ravel() > 0  # 支持向量的类别标签由dual_coef_的正负给出
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)  # 支持向量绘制
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# SVM 调参 C gamma控制高斯核的宽度
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 10))
for ax, C in zip(axes2, [-1, 0, 3]):  # C分别取[0.1, 1, 1000]
    for a, gamma in zip(ax, range(-1, 2)):  # gamma分别取[0.1, 1, 10]
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes2[0, 0].legend(["Class0", "Class1", "sv class 0", "sv clas 1"],
                   ncol=4, loc=(.9, 1.2))  # 添加图例说明类别和支持向量
plt.show()
