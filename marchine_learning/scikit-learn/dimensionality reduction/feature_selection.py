import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,precision_score,recall_score
from tqdm import tqdm   #显示进度
import warnings

# 忽略警告信息（如收敛警告等），保持控制台输出整洁
warnings.filterwarnings('ignore')

# ==========================================
# 1. 数据加载与预处理
# ==========================================
# Load the dataset
data = pd.read_csv('../../data/adult.csv')

# 筛选出对象类型（通常为字符串/类别）的列名
dis_cols = data.select_dtypes(include=["object"]).columns.tolist()

# 定义标签列（目标变量）
label_col = 'income'

# 类别特征编码：将文本类别转换为数值索引
for col in dis_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# 分离特征矩阵 X 和目标向量 y
x = data.drop(columns=[label_col])
y = data[label_col]

# 数据标准化：
# 对特征进行 Z-score 标准化（均值为0，方差为1）
# 这对于逻辑回归（基于距离/梯度）和 PCA（基于方差）非常重要
x = pd.DataFrame(StandardScaler().fit_transform(x),columns=x.columns)

# 划分训练集和测试集 (80% 训练, 20% 测试)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ==========================================
# 2. 定义模型与特征选择策略
# ==========================================
# 待评估的分类模型字典
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# 特征降维/选择的方法列表
# None: 不处理, PCA: 主成分分析, Pearson: 相关系数筛选
# MI_*: 基于模型重要性 (Model Importance) 的筛选
feature_decomposition = ['None', 'PCA', 'Pearson', 'MI_LR', 'MI_DT', 'MI_RF']
results = {}

# ==========================================
# 3. 主循环：遍历特征选择方法 -> 训练 -> 评估
# ==========================================
for fd in tqdm(feature_decomposition):
    # 初始化结果表，行索引为模型名，列为评估指标
    result = pd.DataFrame(columns=['AUC', 'Precision', 'Recall'], index=models.keys())

    # --- 特征工程处理 ---
    if fd == 'None':
        # 基准对照：不进行特征选择
        x_train_fd, x_test_fd = x_train, x_test

    elif fd == 'PCA':
        # 降维：主成分分析
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)  # 降维至10个主成分
        x_train_fd = pca.fit_transform(x_train)
        x_test_fd = pca.transform(x_test)  # 测试集使用相同的投影

    elif fd == 'Pearson':
        # 过滤法：皮尔逊相关系数
        # 计算训练集中各特征与目标变量 y 的相关性
        corr = x_train.join(y_train).corr().iloc[:-1, -1]
        # 筛选绝对相关系数大于 0.1 的特征
        top_features = corr[abs(corr) > 0.1].index
        # 兜底策略：如果筛选后无特征，则默认取前10个
        if len(top_features) ==0:
            top_features = x_train.columns[:10]
        x_train_fd = x_train[top_features]
        x_test_fd = x_test[top_features]

    elif fd == 'MI_RF':
        # 嵌入法：基于随机森林特征重要性
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        importances = clf.feature_importances_
        top_features = np.argsort(importances)[-10:]
        x_train_fd = x_train.iloc[:, top_features]
        x_test_fd = x_test.iloc[:, top_features]

    else:

        print('Feature decomposition error')
        continue

    # --- 模型训练与评估 ---
    for model_name, model in models.items():
        # 在处理后的特征上训练模型
        model.fit(x_train_fd, y_train)

        # 预测类别和概率
        y_pred = model.predict(x_test_fd)
        y_pred_proba = model.predict_proba(x_test_fd)[:, 1]  # 获取正类的概率

        # 计算指标
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # 记录结果 (使用 .loc 确保正确赋值给 DataFrame 的行)
        result.loc[model_name] = [auc, precision, recall]

    # 保存当前特征选择方法下的所有模型结果
    results[fd] = result
