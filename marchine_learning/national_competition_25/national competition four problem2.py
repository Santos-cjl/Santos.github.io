import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, classification_report, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
print("步骤1: 数据加载与预处理...")
df = pd.read_excel('../data/national competition data.xlsx')

# 筛选女胎样本并标注染色体异常状态
df_female = df[df['染色体的非整倍体'].str.contains(r'T13|T18|T21|--', na=True, regex=True)]


def label_abnormal(s):
    abnormal = [0, 0, 0]  # [13号, 18号, 21号]
    if 'T13' in str(s): abnormal[0] = 1
    if 'T18' in str(s): abnormal[1] = 1
    if 'T21' in str(s): abnormal[2] = 1
    return pd.Series(abnormal)


df_female[['Abnormal_13', 'Abnormal_18', 'Abnormal_21']] = df_female['染色体的非整倍体'].apply(label_abnormal)

# 特征选择
features = df_female[['年龄', '孕妇BMI', '原始读段数', 'GC含量', 'X染色体的Z值',
                      '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
                      'X染色体浓度', '被过滤掉读段数的比例']]
targets = df_female[['Abnormal_13', 'Abnormal_18', 'Abnormal_21']]

# 处理缺失值
imputer = KNNImputer(n_neighbors=5)
features_imputed = imputer.fit_transform(features)
features_imputed = pd.DataFrame(features_imputed, columns=features.columns)

# 特征缩放
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled,
    targets,
    test_size=0.2,
    random_state=42
)
print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# 2. 模型训练
print("步骤2: 训练MultiOutputClassifier模型...")
base_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

multi_output_model = MultiOutputClassifier(base_rf, n_jobs=-1)
multi_output_model.fit(X_train, y_train)

# 3. 模型评估与获取预测概率
print("步骤3: 模型评估...")
y_pred = multi_output_model.predict(X_test)
y_pred_proba = multi_output_model.predict_proba(X_test)

print(f"子集准确率 (Subset Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"汉明损失 (Hamming Loss): {hamming_loss(y_test, y_pred):.4f}")

# 4. 特征重要性分析
print("步骤4: 特征重要性分析...")
feature_importances = []
for i, estimator in enumerate(multi_output_model.estimators_):
    importance_df = pd.DataFrame({
        'feature': features.columns,
        'importance': estimator.feature_importances_,
        'chromosome': ['13', '18', '21'][i]
    })
    feature_importances.append(importance_df)

# 合并所有染色体的特征重要性
all_importances = pd.concat(feature_importances)
mean_importances = all_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
print("\n平均特征重要性排名:")
print(mean_importances)

# 5. 精细化阈值设定 - 使用PR曲线和F1分数优化阈值[6](@ref)
print("步骤5: 精细化阈值设定...")


def find_optimal_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, f1_scores[optimal_idx]


# 为每个染色体寻找最佳阈值
optimal_thresholds = {}
for idx, chrom in enumerate(['13', '18', '21']):
    y_true_chrom = y_test.iloc[:, idx]
    y_proba_chrom = y_pred_proba[idx][:, 1]
    opt_thresh, opt_f1 = find_optimal_threshold(y_true_chrom, y_proba_chrom)
    optimal_thresholds[chrom] = opt_thresh
    print(f"染色体{chrom}号 - 最佳阈值: {opt_thresh:.4f}, 对应F1分数: {opt_f1:.4f}")

# 6. 构建综合数据质量评分函数
print("步骤6: 构建综合数据质量评分函数...")


def calculate_data_quality_score(row):
    """
    计算综合数据质量评分（0-1之间，越高表示质量越好）
    """
    # 1. 被过滤读段比例越低越好
    filtered_reads_score = 1 - min(row['被过滤掉读段数的比例'] / 0.1, 1.0)

    # 2. 原始读段数越高越好
    total_reads_score = min(row['原始读段数'] / 1e7, 1.0)

    # 3. GC含量在正常范围内越好
    gc_content = row['GC含量']
    if 35 <= gc_content <= 65:
        gc_score = 1.0
    else:
        gc_score = 1 - (min(abs(gc_content - 35), abs(gc_content - 65)) / 50)

    # 4. X染色体Z值绝对值越低越好，表明技术误差小
    x_zscore_score = 1 - min(abs(row['X染色体的Z值']) / 5.0, 1.0)

    # 赋予权重并计算综合评分
    weights = [0.3, 0.3, 0.2, 0.2]
    total_score = (filtered_reads_score * weights[0] +
                   total_reads_score * weights[1] +
                   gc_score * weights[2] +
                   x_zscore_score * weights[3])

    return max(0, min(total_score, 1))


# 7. 综合判定函数
print("步骤7: 构建改进的判定函数...")


def comprehensive_judgment_advanced(row):
    """
    改进的综合判定函数
    使用动态阈值和多重证据进行判定
    """
    # 计算该样本的数据质量评分
    dq_score = calculate_data_quality_score(row)

    results = []
    chromosomes = ['13', '18', '21']
    prob_columns = [f'Pred_Prob_{chrom}' for chrom in chromosomes]
    z_columns = [f'{chrom}号染色体的Z值' for chrom in chromosomes]

    for chrom, prob_col, z_col in zip(chromosomes, prob_columns, z_columns):
        base_thresh = optimal_thresholds[chrom]
        # 根据数据质量动态调整阈值
        dynamic_threshold = base_thresh + (1 - dq_score) * 0.2
        dynamic_threshold = min(dynamic_threshold, 0.9)

        prob = row[prob_col]
        z_val = abs(row[z_col])

        # 判定逻辑：综合考虑预测概率、Z值和质量评分
        if prob > dynamic_threshold:
            if z_val > 3.0:
                risk_level = "高风险"
                results.append(f"{chrom}号染色体{risk_level}")
            elif z_val > 2.0:
                risk_level = "中等风险"
                results.append(f"{chrom}号染色体{risk_level}，建议复查")
            else:
                results.append(f"{chrom}号染色体预测概率高({prob:.3f})，但Z值证据不足({z_val:.2f})，建议进一步确认")
        elif z_val > 3.0 and dq_score > 0.6:
            results.append(f"{chrom}号染色体Z值异常({z_val:.2f})，建议人工审核")

    if results:
        if dq_score < 0.5:
            return "； ".join(results) + " (注：数据质量较差，结果可靠性可能降低)"
        else:
            return "； ".join(results)
    else:
        return "低风险"


# 8. 应用判定函数到测试集
print("步骤8: 应用判定函数到测试集...")
# 将预测概率添加到测试集DataFrame中
X_test_with_probs = X_test.copy()
for idx, chrom in enumerate(['13', '18', '21']):
    X_test_with_probs[f'Pred_Prob_{chrom}'] = y_pred_proba[idx][:, 1]

# 应用改进的判定函数
X_test_with_probs['综合判定结果'] = X_test_with_probs.apply(comprehensive_judgment_advanced, axis=1)

# 添加真实标签和预测标签
X_test_with_probs['Abnormal_13_true'] = y_test['Abnormal_13'].values
X_test_with_probs['Abnormal_18_true'] = y_test['Abnormal_18'].values
X_test_with_probs['Abnormal_21_true'] = y_test['Abnormal_21'].values
X_test_with_probs['Abnormal_13_pred'] = y_pred[:, 0]
X_test_with_probs['Abnormal_18_pred'] = y_pred[:, 1]
X_test_with_probs['Abnormal_21_pred'] = y_pred[:, 2]

# 9. 可视化 - 分别生成四个图表
print("步骤9: 分别生成可视化图表...")

# 设置全局字体大小
plt.rcParams['font.size'] = 12

# 9.1 特征重要性可视化 - 单独保存
plt.figure(figsize=(10, 8))
top_features = mean_importances.head(10)
bars = plt.barh(range(len(top_features)), top_features.values, color='skyblue')
plt.yticks(range(len(top_features)), top_features.index)
plt.xlabel('平均重要性')
plt.title('Top 10 平均特征重要性排名')
plt.grid(True, alpha=0.3, axis='x')

# 在条形图上添加数值标签
for i, (feature, importance) in enumerate(zip(top_features.index, top_features.values)):
    plt.text(importance + 0.001, i, f'{importance:.4f}', va='center')

plt.tight_layout()
plt.savefig('feature_importance_ranking.png', dpi=300, bbox_inches='tight')
plt.show()
print("已保存: feature_importance_ranking.png")


# 9.2 数据质量分布 - 单独保存
plt.figure(figsize=(10, 6))
dq_scores = X_test_with_probs.apply(calculate_data_quality_score, axis=1)
n, bins, patches = plt.hist(dq_scores, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')

# 添加数值标签
for i, (count, bin_edge) in enumerate(zip(n, bins)):
    if count > 0:
        plt.text(bin_edge + (bins[1] - bins[0])/2, count + 0.5, str(int(count)), ha='center')

plt.xlabel('数据质量评分')
plt.ylabel('频数')
plt.title('测试集数据质量分布')
plt.axvline(x=0.5, color='r', linestyle='--', label='质量合格线(0.5)')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('data_quality_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("已保存: data_quality_distribution.png")

# 10. 保存结果
print("步骤10: 保存结果...")
X_test_with_probs.to_csv('chromosome_abnormal_prediction_advanced_results.csv', index=False, encoding='utf-8-sig')

# 输出最终总结
print("\n=== 分析完成 ===")
print(f"1. 模型准确率指标:")
print(f"   - 子集准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"   - 汉明损失: {hamming_loss(y_test, y_pred):.4f}")

print(f"\n2. 各染色体优化阈值:")
for chrom in ['13', '18', '21']:
    print(f"   - 染色体{chrom}号: {optimal_thresholds[chrom]:.4f}")

print(f"\n3. 判定结果分布:")
result_distribution = X_test_with_probs['综合判定结果'].value_counts()
for result, count in result_distribution.items():
    print(f"   - {result}: {count}例 ({(count / len(X_test_with_probs)):.1%})")

    print("\n详细结果已保存至: chromosome_abnormal_prediction_advanced_results.csv")
    print("可视化图表已保存至: chromosome_analysis_results.png")

# 1. 首先，创建原始数据的一个副本，以防万一需要回滚
X_test_original = X_test_with_probs.copy()
print(f"删除前总样本数: {len(X_test_with_probs)}")

# 2. 精确识别健康人群，集所有染色体真实标签均为0的样本
healthy_criteria = (X_test_with_probs['Abnormal_13_true'] == 0) & \
                   (X_test_with_probs['Abnormal_18_true'] == 0) & \
                   (X_test_with_probs['Abnormal_21_true'] == 0)
healthy_population = X_test_with_probs[healthy_criteria]
print(f"识别出的健康人群样本数量: {len(healthy_population)}")

# 3. 从数据集中删除健康人群
# 保留那些“非全健康”的样本
X_test_non_healthy = X_test_with_probs[~healthy_criteria]
print(f"删除健康人群后剩余样本数: {len(X_test_non_healthy)}")

# 4. 验证删除操作是否正确
# 检查剩余数据中是否确实不再包含“全健康”的样本
remaining_healthy_check = (X_test_non_healthy['Abnormal_13_true'] == 0) & \
                           (X_test_non_healthy['Abnormal_18_true'] == 0) & \
                           (X_test_non_healthy['Abnormal_21_true'] == 0)
print(f"删除操作后，剩余数据中是否意外包含健康样本: {remaining_healthy_check.any()}")
# 如果输出为False，说明删除成功；如果为True，说明删除逻辑有误，需要检查。

# 5. 保存处理后的数据集，仅包含非健康样本
X_test_non_healthy.to_csv('test_set_only_abnormal_samples.csv', index=False, encoding='utf-8-sig')
print(f"\n处理完成! 仅包含异常样本的数据已保存至: test_set_only_abnormal_samples.csv")
print(f"原始健康样本数量: {len(healthy_population)}")
print(f"最终分析的异常样本总数: {len(X_test_non_healthy)}")

# 1. 定义新风险等级分类函数
def categorize_risk(detailed_result):
    """
    根据综合判定结果字符串分类风险等级
    """
    if '高风险' in detailed_result:
        return '高风险'
    elif '中等风险' in detailed_result or '建议复查' in detailed_result:
        return '中风险'
    elif '低风险' in detailed_result:
        return '低风险'
    else:
        return '未知'

# 2. 应用风险等级分类
X_test_non_healthy['风险等级'] = X_test_non_healthy['综合判定结果'].apply(categorize_risk)

# 3. 计算各风险等级的样本数量
risk_level_counts = X_test_non_healthy['风险等级'].value_counts()
print("各风险等级样本数量分布:")
print(risk_level_counts)

# 4. 准备可视化数据
# 4.1 风险等级总体分布
risk_categories = ['高风险', '中风险', '低风险']
risk_counts = [risk_level_counts.get('高风险', 0),
               risk_level_counts.get('中风险', 0),
               risk_level_counts.get('低风险', 0)]

# 4.2 各风险等级下染色体异常的详细分布
abnormal_details = {}
for level in ['高风险', '中风险', '低风险']:
    subset = X_test_non_healthy[X_test_non_healthy['风险等级'] == level]
    abnormal_details[level] = {
        '13号异常': subset['Abnormal_13_true'].sum(),
        '18号异常': subset['Abnormal_18_true'].sum(),
        '21号异常': subset['Abnormal_21_true'].sum(),
        '总样本数': len(subset)
    }

# 5. 创建可视化图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 5.1 风险等级总体分布饼图
colors = ['#FF6B6B', '#FFD93D', '#4ECDC4']  # 红-黄-绿
wedges, texts, autotexts = ax1.pie(risk_counts, labels=risk_categories, autopct='%1.1f%%',
                                    colors=colors, startangle=90, explode=(0.05, 0.05, 0.05))
ax1.set_title('异常样本风险等级分布', fontsize=16, fontweight='bold')

# 5.2 各风险等级下染色体异常详细分布堆叠柱状图
x = np.arange(len(risk_categories))
width = 0.6

# 计算各染色体异常数量
abnormal_13_counts = [abnormal_details[level]['13号异常'] for level in risk_categories]
abnormal_18_counts = [abnormal_details[level]['18号异常'] for level in risk_categories]
abnormal_21_counts = [abnormal_details[level]['21号异常'] for level in risk_categories]
total_counts = [abnormal_details[level]['总样本数'] for level in risk_categories]

# 绘制堆叠柱状图
bars1 = ax2.bar(x, abnormal_13_counts, width, label='13号异常', color='#FF9AA2')
bars2 = ax2.bar(x, abnormal_18_counts, width, bottom=abnormal_13_counts, label='18号异常', color='#FFB347')
bars3 = ax2.bar(x, abnormal_21_counts, width,
                bottom=np.array(abnormal_13_counts) + np.array(abnormal_18_counts),
                label='21号异常', color='#6A0572')

ax2.set_xlabel('风险等级', fontsize=12)
ax2.set_ylabel('样本数量', fontsize=12)
ax2.set_title('各风险等级下染色体异常详细分布', fontsize=16, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(risk_categories)
ax2.legend()

# 在柱子上添加总数标签
for i, total in enumerate(total_counts):
    ax2.text(i, total + 0.5, f'{total}', ha='center', va='bottom', fontweight='bold')

# 添加网格线
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

# 6. 添加表格展示详细数据
table_data = []
for level in risk_categories:
    data = abnormal_details[level]
    table_data.append([
        level,
        data['总样本数'],
        data['13号异常'],
        data['18号异常'],
        data['21号异常'],
        f"{data['13号异常']/data['总样本数']*100:.1f}%" if data['总样本数'] > 0 else "0%",
        f"{data['18号异常']/data['总样本数']*100:.1f}%" if data['总样本数'] > 0 else "0%",
        f"{data['21号异常']/data['总样本数']*100:.1f}%" if data['总样本数'] > 0 else "0%"
    ])

# 创建表格
table = plt.table(
    cellText=table_data,
    colLabels=['风险等级', '总样本数', '13号异常', '18号异常', '21号异常', '13号占比', '18号占比', '21号占比'],
    loc='bottom',
    bbox=[0, -0.5, 1, 0.3]  # [left, bottom, width, height]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# 调整布局
plt.subplots_adjust(bottom=0.3)

# 7. 保存和显示图表
plt.tight_layout()
plt.savefig('risk_level_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 输出详细统计结果
print("\n=== 详细统计结果 ===")
for level in risk_categories:
    data = abnormal_details[level]
    print(f"\n{level}样本详细分析:")
    print(f"  总样本数: {data['总样本数']}")
    print(f"  13号染色体异常: {data['13号异常']}例 ({data['13号异常']/data['总样本数']*100:.1f}%)")
    print(f"  18号染色体异常: {data['18号异常']}例 ({data['18号异常']/data['总样本数']*100:.1f}%)")
    print(f"  21号染色体异常: {data['21号异常']}例 ({data['21号异常']/data['总样本数']*100:.1f}%)")

# 9. 计算并输出模型整体性能指标
total_abnormal = sum([abnormal_details[level]['13号异常'] + abnormal_details[level]['18号异常'] +
                      abnormal_details[level]['21号异常'] for level in risk_categories])
correctly_identified = abnormal_details['高风险']['13号异常'] + abnormal_details['高风险']['18号异常'] + \
                      abnormal_details['高风险']['21号异常'] + abnormal_details['中风险']['13号异常'] + \
                      abnormal_details['中风险']['18号异常'] + abnormal_details['中风险']['21号异常']

print(f"\n模型整体性能:")
print(f"  总异常样本数: {total_abnormal}")
print(f"  被正确识别为有风险的异常样本数: {correctly_identified}")
print(f"  模型敏感度: {correctly_identified/total_abnormal*100:.1f}%")


#模型的敏感度分析
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt

# 1. 定义问题，确定输入变量及其范围
# 模型有以下特征，根据训练数据的实际范围定义边界
problem = {
    'num_vars': 10,
    'names': ['年龄', '孕妇BMI', '原始读段数', 'GC含量', 'X染色体的Z值',
              '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
              'X染色体浓度', '被过滤掉读段数的比例'],
    'bounds': [
        [20, 40],  # 年龄的近似范围
        [18, 35],  # BMI的近似范围
        [1e6, 1e8],  # 原始读段数范围
        [0.35, 0.65],  # GC含量范围
        [-5.0, 5.0],  # X染色体Z值范围
        [-5.0, 5.0],  # 13号染色体Z值范围
        [-5.0, 5.0],  # 18号染色体Z值范围
        [-5.0, 5.0],  # 21号染色体Z值范围
        [0.0, 0.5],  # X染色体浓度范围
        [0.0, 0.1]  # 被过滤读段比例范围
    ]
}

# 2. 生成样本
# N值计算：Sobol方法所需样本数 = N * (2D + 2), 其中D是变量数，N是基础样本数。
param_values = saltelli.sample(problem, N=500, calc_second_order=False)  # N为基础样本数，calc_second_order=False忽略二阶交互以节省计算

print(f"生成了 {param_values.shape[0]} 个样本用于敏感度分析")


# 3. 封装模型
# 创建一个函数，接收SALib生成的样本，返回模型的预测
def model_wrapper(params):
    """
    将SALib生成的参数样本转换为你的模型所需的输入格式并进行预测。
    这里以预测21号染色体异常的概率为例。
    """
    # 将参数数组转换为DataFrame，列名与训练时一致
    X_test = pd.DataFrame(params, columns=problem['names'])

    # X_test_scaled = scaler.transform(X_test)

    # 使用训练好的模型进行预测
    # 注意：SALib要求模型输出为一维数组，所以我们选取21号染色体异常的概率作为输出目标
    y_pred_prob = multi_output_model.predict_proba(X_test)  # 如果使用预处理：X_test_scaled
    # 获取21号染色体异常的预测概率（假设是第三个估计器，索引为2）
    chromosome_21_proba = y_pred_prob[2][:, 1]  # [:,1] 获取正例（异常）的概率

    return chromosome_21_proba


# 4. 在所有样本上运行模型
Y = model_wrapper(param_values)

# 5. 执行Sobol分析
Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=True)

# 6. 可视化结果
plt.figure(figsize=(10, 6))
plt.bar(problem['names'], Si['ST'], color='skyblue', edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.title('Sobol总效应敏感度指数（针对21号染色体异常概率）')
plt.ylabel('总效应指数 (ST)')
plt.tight_layout()
plt.show()

# 7. 输出详细结果
print("\nSobol敏感度指数分析结果（21号染色体异常）：")
for i, name in enumerate(problem['names']):
    print(f"特征 '{name}':")
    print(f"  一阶指数 (S1): {Si['S1'][i]:.4f}")  # 主效应，衡量该变量独自对输出方差的贡献
    print(f"  总效应指数 (ST): {Si['ST'][i]:.4f}")  # 总效应，包括该变量及其与所有其他变量的交互作用对输出方差的贡献
    print()

