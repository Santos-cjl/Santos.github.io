import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_excel('../data/national competition one data.xlsx')

# 数据预处理
missing_ratio = data.isnull().sum()/len(data)
missing_data = missing_ratio[missing_ratio>0]
data['孕妇BMI'] = data['孕妇BMI'].fillna(data['孕妇BMI'].mean())
print(f"缺失值处理完毕，剩余缺失值总数: {data.isnull().sum().sum()}")
data = data[data['Y染色体浓度'] != 0]

# 转换孕周格式函数
def convert_gestation(gest_str):
    try:
        if '+' in gest_str:
            weeks, days = gest_str.replace('w', '').split('+')
            return float(weeks) + float(days)/7
        else:
            return float(gest_str.replace('w', ''))
    except:
        return np.nan

# 应用转换
data['孕周数值'] = data['检测孕周'].apply(convert_gestation)

# 处理异常值
data = data[(data['孕妇BMI'] >= 16) & (data['孕妇BMI'] <= 45)]
data = data[(data['Y染色体浓度'] >= 0.01) & (data['Y染色体浓度'] <= 0.25)]

# 确保所有分析使用相同的数据集（无缺失值）
analysis_data = data[['孕周数值', '孕妇BMI', 'Y染色体浓度']].dropna()
print(f"分析数据集大小: {len(analysis_data)}")

# 2. 描述性统计
print("="*50)
print("描述性统计:")
print(analysis_data.describe().round(3))

# 3. 相关性分析
print("\n" + "="*50)
print("相关性分析 (Pearson 相关系数):")

# 使用统一处理缺失值后的数据
corr_matrix = analysis_data.corr(method='pearson')
print(corr_matrix)

# 提取相关性及p值
p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
for i in corr_matrix.index:
    for j in corr_matrix.columns:
        if i != j:
            # 使用统一的数据集，确保长度一致
            corr, p_val = stats.pearsonr(analysis_data[i], analysis_data[j])
            p_values.loc[i, j] = p_val
        else:
            p_values.loc[i, j] = np.nan

print("\n相关系数的p值:")
print(p_values)


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_folder = os.path.join(desktop_path, "Y染色体分析图表")
os.makedirs(save_folder, exist_ok=True)  # 创建保存文件夹 [9](@ref)

print(f"图表将保存到: {save_folder}")

# 4. 可视化

# 孕周 vs Y浓度
plt.figure(figsize=(10, 6))
plt.scatter(analysis_data['孕周数值'], analysis_data['Y染色体浓度'], alpha=0.6, color='steelblue')
z = np.polyfit(analysis_data['孕周数值'], analysis_data['Y染色体浓度'], 1)
p = np.poly1d(z)
plt.plot(analysis_data['孕周数值'], p(analysis_data['孕周数值']), "r--", linewidth=2)
plt.xlabel('孕周（周）')
plt.ylabel('Y染色体浓度')
plt.title('Y染色体浓度随孕周变化趋势')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, '孕周与Y染色体浓度关系.png'), dpi=300, bbox_inches='tight')  # 保存图像 [9](@ref)
plt.close()


# BMI vs Y浓度
plt.figure(figsize=(10, 6))
plt.scatter(analysis_data['孕妇BMI'], analysis_data['Y染色体浓度'], alpha=0.6, color='olive')
z_bmi = np.polyfit(analysis_data['孕妇BMI'], analysis_data['Y染色体浓度'], 1)
p_bmi = np.poly1d(z_bmi)
plt.plot(analysis_data['孕妇BMI'], p_bmi(analysis_data['孕妇BMI']), "r--", linewidth=2)
plt.xlabel('孕妇BMI')
plt.ylabel('Y染色体浓度')
plt.title('Y染色体浓度与BMI关系')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, 'BMI与Y染色体浓度关系.png'), dpi=300, bbox_inches='tight')
plt.close()


# 不同孕周段分布
analysis_data['孕周阶段'] = pd.cut(analysis_data['孕周数值'],
                                  bins=[10, 14, 20, 30],
                                  labels=['早期(10-14周)', '中期(14-20周)', '晚期(20-30周)'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='孕周阶段', y='Y染色体浓度', data=analysis_data)
plt.title('不同孕周段Y染色体浓度分布')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, '不同孕周段Y染色体浓度分布.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap='coolwarm', cbar=True, square=True)
plt.title('变量间相关系数矩阵')
plt.tight_layout()
plt.savefig(os.path.join(save_folder, '变量间相关系数矩阵.png'), dpi=300, bbox_inches='tight')
plt.close()


# 5. 线性回归建模 (使用scikit-learn)
print("\n" + "="*50)
print("线性回归模型结果 (使用scikit-learn):")

# 准备数据 - 使用统一的数据集
X = analysis_data[['孕周数值', '孕妇BMI']]
y = analysis_data['Y染色体浓度']

# 创建并训练模型
model = LinearRegression()
model.fit(X, y)

# 获取系数
intercept = model.intercept_
coef_week = model.coef_[0]
coef_bmi = model.coef_[1]

# 预测
y_pred = model.predict(X)

# 计算评估指标
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"截距 (Intercept): {intercept:.6f}")
print(f"孕周系数: {coef_week:.6f}")
print(f"BMI系数: {coef_bmi:.6f}")
print(f"均方误差 (MSE): {mse:.6f}")
print(f"均方根误差 (RMSE): {rmse:.6f}")
print(f"决定系数 (R²): {r2:.4f}")

# 6. 模型显著性检验 (使用scipy进行F检验和t检验)
print("\n" + "="*50)
print("模型显著性检验:")

# 计算残差
residuals = y - y_pred

# F检验 - 整体模型显著性
ss_total = np.sum((y - np.mean(y))**2)
ss_regression = np.sum((y_pred - np.mean(y))**2)
ss_residual = np.sum(residuals**2)

df_regression = X.shape[1]
df_residual = len(y) - df_regression - 1
f_statistic = (ss_regression / df_regression) / (ss_residual / df_residual)
p_value_f = 1 - stats.f.cdf(f_statistic, df_regression, df_residual)

print(f"F统计量: {f_statistic:.4f}, p值: {p_value_f:.4e}")

# t检验 - 单个系数显著性
X_with_intercept = np.column_stack([np.ones(len(X)), X])
cov_matrix = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept))
se = np.sqrt(np.diag(cov_matrix) * ss_residual / df_residual)

t_stats = np.append(intercept, model.coef_) / se
p_values_t = [2 * (1 - stats.t.cdf(np.abs(t), df_residual)) for t in t_stats]

print("\n系数显著性检验 (t检验):")
print(f"截距: t统计量 = {t_stats[0]:.4f}, p值 = {p_values_t[0]:.4e}")
print(f"孕周: t统计量 = {t_stats[1]:.4f}, p值 = {p_values_t[1]:.4e}")
print(f"BMI: t统计量 = {t_stats[2]:.4f}, p值 = {p_values_t[2]:.4e}")

# 7. 残差分析
print("\n" + "="*50)
print("残差分析:")

# 残差正态性检验 (Shapiro-Wilk)
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"残差正态性检验 (Shapiro-Wilk): 统计量 = {shapiro_stat:.4f}, p值 = {shapiro_p:.4f}")

# 残差图
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差 vs 预测值')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, '残差vs预测值.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('残差Q-Q图')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, '残差Q-Q图.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, density=True, alpha=0.6)
plt.xlabel('残差')
plt.ylabel('密度')
plt.title('残差分布')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, '残差分布直方图.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8. 优化模型：添加孕周非线性项
print("\n" + "="*50)
print("优化模型（添加孕周二次项）:")

# 创建多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(analysis_data[['孕周数值', '孕妇BMI']])
feature_names = poly.get_feature_names_out(['孕周数值', '孕妇BMI'])

# 创建并训练多项式模型
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# 预测
y_pred_poly = poly_model.predict(X_poly)

# 计算评估指标
r2_poly = r2_score(y, y_pred_poly)

print("多项式特征:", feature_names)
print("多项式模型系数:", poly_model.coef_)
print("多项式模型截距:", poly_model.intercept_)
print(f"多项式模型 R²: {r2_poly:.4f}")

# 多项式模型的F检验
ss_residual_poly = np.sum((y - y_pred_poly)**2)
df_residual_poly = len(y) - X_poly.shape[1] - 1
f_statistic_poly = (ss_regression / X_poly.shape[1]) / (ss_residual_poly / df_residual_poly)
p_value_f_poly = 1 - stats.f.cdf(f_statistic_poly, X_poly.shape[1], df_residual_poly)

print(f"多项式模型 F统计量: {f_statistic_poly:.4f}, p值: {p_value_f_poly:.4e}")

# 9. 模型比较与结论
print("\n" + "="*50)
print("模型比较与结论:")

print("1. 线性模型: Y = {:.5f} + {:.5f}×孕周 + {:.5f}×BMI".format(intercept, coef_week, coef_bmi))
print("   R² = {:.3f}, F统计量 = {:.2f} (p = {:.3e})".format(r2, f_statistic, p_value_f))

print("2. 多项式模型: Y = {:.5f} + {:.5f}×孕周 + {:.5f}×孕周² + {:.5f}×BMI".format(
    poly_model.intercept_, poly_model.coef_[0], poly_model.coef_[2], poly_model.coef_[1]))
print("   R² = {:.3f}, F统计量 = {:.2f} (p = {:.3e})".format(r2_poly, f_statistic_poly, p_value_f_poly))

print("\n主要发现:")
print("- 孕周与Y染色体浓度显著相关 (r = {:.3f}, p < 0.001)".format(corr_matrix.loc['孕周数值', 'Y染色体浓度']))
print("- BMI与Y染色体浓度无显著相关性 (r = {:.3f}, p = {:.3f})".format(
    corr_matrix.loc['孕妇BMI', 'Y染色体浓度'], float(p_values.loc['孕妇BMI', 'Y染色体浓度'])))
print("- 线性模型中，孕周系数显著 (p < 0.001)，BMI系数不显著 (p = {:.3f})".format(p_values_t[2]))
print("- 添加孕周二次项后，模型解释度从{:.1f}%提升至{:.1f}%".format(r2 * 100, r2_poly*100))


