import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']  # 指定默认字体（支持中文）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据来源：图片2/4 表4-4
years = [1950, 1955, 1960, 1966, 1970, 1975, 1980]
local_export_ratio = [11.0, 28.8, 72.8, 75.8, 81.0, 76.6, 69.4] # 单位：%

plt.figure(figsize=(10,6))
plt.plot(years, local_export_ratio, marker='o', linewidth=2, markersize=8)
plt.axvline(x=1966, color='grey', linestyle='--', alpha=0.7, label='转型关键节点 (~1966)')
plt.xlabel('年份')
plt.ylabel('本港产品出口比重 (%)')
plt.title('图1：香港从转口贸易向出口加工转型 (1950-1980)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('图1.png',dpi=300,bbox_inches='tight')
plt.show()

# 数据来源：图片4 表4-5
years = [1971, 1975, 1985, 1988, 1993, 1998, 2000]
manufacturing_share = [28.1, 26.9, 22.0, 20.4, 11.1, 6.0, 5.8] # 制造业占比 %
wholesale_retail_share = [19.5, 20.7, 22.8, 25.0, 27.1, 25.0, 26.4] # 批发零售等占比 %
finance_share = [17.5, 17.0, 16.1, 19.0, 26.1, 24.6, 23.7] # 金融保险等占比 %

plt.figure(figsize=(12,7))
plt.plot(years, manufacturing_share, marker='s', label='制造业', linewidth=2)
plt.plot(years, wholesale_retail_share, marker='o', label='批发零售、酒店餐饮业', linewidth=2)
plt.plot(years, finance_share, marker='^', label='金融保险地产及商业服务', linewidth=2)
plt.axvline(x=1988, color='red', linestyle='--', alpha=0.7, label='服务业超越制造业 (1988)')
plt.xlabel('年份')
plt.ylabel('占GDP比重 (%)')
plt.title('图2：香港产业结构变迁：制造业的衰落与服务业的崛起 (1971-2000)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('图2.png',dpi=300,bbox_inches='tight')
plt.show()

# 概念示意图：新加坡产业升级的阶梯
stages = ['劳动密集型\n(1960s-1970s)', '资本/技术密集型\n(1980s-1990s)', '知识/创新密集型\n(2000s-)']
representative_industries = ['假发、纺织、玩具\n简单电子装配', '石化、精密机械\n硬盘驱动器、半导体', '生物医药、金融、总部经济\n水务、教育、医疗中心']
gov_role = ['创造就业\n招商引资(轻工业)', '战略引导\n基础设施/教育投入', '前瞻布局\n全球网络/创新生态']

fig, ax = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(stages))
bar_height = 0.6

# 绘制概念柱状图（用颜色深度代表产业复杂度/附加值）
colors = ['lightblue', 'cornflowerblue', 'royalblue']
bars = ax.barh(y_pos, [1, 1.5, 2], height=bar_height, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(stages, fontsize=12)
ax.invert_yaxis()  # 从上到下阅读
ax.set_xlabel('产业复杂度 / 附加值', fontsize=12)
ax.set_title('图4：新加坡政府引导下的产业阶梯式升级路径', fontsize=14, pad=20)

# 在条形图上添加文本标注
for i, (bar, industry, role) in enumerate(zip(bars, representative_industries, gov_role)):
    width = bar.get_width()
    ax.text(width/2, bar.get_y() + bar.get_height()/2, industry + '\n(政府角色: ' + role + ')',
            ha='center', va='center', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('图3.png',dpi=300,bbox_inches='tight')
plt.show()

# 概念图：双路径模型
# 此图描绘两种不同的升级路径
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 新加坡路径：阶梯式、主动
ax1 = axes[0]
x_sg = [0, 1, 2, 3]
y_sg = [1, 2, 3, 4]  # 更陡峭的升级
ax1.plot(x_sg, y_sg, marker='o', linewidth=3, markersize=10, label='产业升级路径')
ax1.fill_between(x_sg, 0, y_sg, alpha=0.1)
ax1.set_xticks(x_sg)
ax1.set_xticklabels(['劳动力密集\n(1960s)', '资本技术密集\n(1980s)', '知识创新密集\n(1990s)', '全球网络/总部\n(2000s+)'], fontsize=10)
ax1.set_ylabel('产业附加值/复杂度')
ax1.set_title('新加坡：政府引领的主动阶梯式升级')
ax1.text(0.5, 1.5, '政府规划\n目标明确', style='italic', bbox={'facecolor': 'yellow', 'alpha': 0.3})
ax1.text(1.5, 2.5, '战略招商\n基建先行', style='italic', bbox={'facecolor': 'yellow', 'alpha': 0.3})
ax1.grid(True, alpha=0.3)
ax1.legend()

# 香港路径：平滑、市场驱动
ax2 = axes[1]
x_hk = [0, 1, 2, 3]
y_hk = [1, 1.8, 2.5, 4]  # 更平滑，后期跳跃
ax2.plot(x_hk, y_hk, marker='s', linewidth=3, markersize=10, label='产业升级路径', color='green')
ax2.fill_between(x_hk, 0, y_hk, alpha=0.1, color='green')
ax2.set_xticks(x_hk)
ax2.set_xticklabels(['转口贸易\n(1950前)', '出口加工\n(1960s)', '技术密集型制造\n(1980s)', '金融服务业主导\n(1990s+)'], fontsize=10)
ax2.set_ylabel('产业附加值/复杂度')
ax2.set_title('香港：市场驱动的被动适应型转型')
ax2.text(0.8, 1.2, '外部冲击\n被迫转型', style='italic', bbox={'facecolor': 'lightgreen', 'alpha': 0.3})
ax2.text(2.2, 2.8, '成本压力\n产业北移', style='italic', bbox={'facecolor': 'lightgreen', 'alpha': 0.3})
ax2.text(3.2, 3.5, '自发转向\n金融服务', style='italic', bbox={'facecolor': 'lightgreen', 'alpha': 0.3})
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle('图5：新加坡与香港经济结构变迁路径比较模型', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig('图4.png',dpi=300,bbox_inches='tight')
plt.show()