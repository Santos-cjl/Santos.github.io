# Machine Learning 学习与项目目录

本目录记录从 Python 数据分析基础到机器学习建模、再到数学建模竞赛实践的学习路线。

| 目录 | 内容 |
|---|---|
| [numpy_pandas](./numpy_pandas) | NumPy 数组与矩阵；Pandas 筛选、分组、合并和绘图 |
| [seaborn](./seaborn) | Matplotlib/Seaborn 图表、热力图和经济学案例 |
| [python_data_analyse](./python_data_analyse) | Python 基础、投资组合、汇率与股票收益率、OLS 回归 |
| [scikit-learn](./scikit-learn) | SVM、决策树、回归、PCA、聚类、集成学习和神经网络 |
| [national_competition_25](./national_competition_25) | 2025 年全国大学生数学建模竞赛 NIPT 数据建模项目 |

## 学习内容

`numpy_pandas` 和 `seaborn` 用于掌握数组、DataFrame、分组连接、分布图、箱线图和热力图。

`python_data_analyse` 将基础语法用于真实数据，计算收益率、组合权重、协方差和因子得分，并分析汇率与股票时间序列。

`scikit-learn` 包含：

- SVM 的决策边界、核函数及 `C`/`gamma` 调参；
- 决策树、线性/多项式回归、岭回归、Lasso 和弹性网；
- PCA、EDA、过滤法、嵌入法和包装法；
- K-Means、GMM、DBSCAN 和层次聚类；
- 集成学习、Stacking、SMOTE 和类别不平衡；
- TensorFlow 全连接网络、早停和学习率衰减。

## 数学建模竞赛项目

`national_competition_25` 对应 2025 年全国大学生数学建模竞赛，包含 NIPT 数据清洗、相关性分析、回归、残差检验、13/18/21 号染色体多标签随机森林分类、PR/F1 阈值优化、数据质量评分和风险可视化。

> 该数据涉及产前筛查，仅用于课程、竞赛和算法练习，不能替代临床诊断。公开或复用数据时必须遵守授权、隐私和伦理要求。

## 运行提示

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn statsmodels xgboost imbalanced-learn openpyxl jupyter tensorflow mglearn sklearn-lvq
```

部分竞赛脚本使用 `../data/national competition data.xlsx` 等路径，原始工作簿不一定包含在仓库中；房价案例使用 `fetch_openml`，首次运行需要网络。图表和预测 CSV 建议保存到独立输出目录。不要上传密钥、个人隐私、受保护医疗数据或竞赛保密资料。

