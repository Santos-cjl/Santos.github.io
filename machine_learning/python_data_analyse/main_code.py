import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import matplotlib

##一、主代码部分
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取人民币汇率数据
fx = pd.read_excel("2021.5.10-2026.5.8人民币汇率中间价.xlsx")
fx = fx.rename(columns={"日期":"date","USD/CNY":"exchange_rate"})

# 清理日期列和数值列，确保都是有效数据
fx = fx[pd.to_datetime(fx['date'], errors='coerce').notna()]
fx['date'] = pd.to_datetime(fx['date'])
fx['exchange_rate'] = pd.to_numeric(fx['exchange_rate'], errors='coerce')
fx = fx.dropna(subset=['exchange_rate'])
fx = fx.sort_values("date")

# 计算汇率变化率
fx['exchange_rate_return'] = fx['exchange_rate'].pct_change()

# 读取传音控股股价数据
stock = pd.read_excel("传音控股股价.xlsx")
stock = stock.rename(columns={"date":"date","close":"stock_price"})

# 清理日期和股价列
stock['date'] = pd.to_datetime(stock['date'])
stock['stock_price'] = pd.to_numeric(stock['stock_price'], errors='coerce')
stock = stock.dropna(subset=['stock_price'])
stock = stock.sort_values("date")

# 计算股票收益率
stock['stock_return'] = stock['stock_price'].pct_change()

# 按日期合并两个数据集
df = pd.merge(fx[['date','exchange_rate_return']],
              stock[['date','stock_return']],
              on='date', how='inner')

# 相关性分析（皮尔逊相关系数）
correlation = df[['exchange_rate_return','stock_return']].corr()
print("相关系数矩阵：\n", correlation)

# 可视化
plt.figure(figsize=(12,5))

# 时间序列图：汇率变化率 vs 股票收益率
plt.subplot(1,2,1)
plt.plot(df['date'], df['exchange_rate_return'], label='Exchange Rate Change')
plt.plot(df['date'], df['stock_return'], label='Stock Return')
plt.legend()
plt.title("Exchange Rate Change vs Stock Return")
plt.xlabel("Date")
plt.ylabel("Return")

# 散点图：观察相关性
plt.subplot(1,2,2)
plt.scatter(df['exchange_rate_return'], df['stock_return'])
plt.xlabel("Exchange Rate Change")
plt.ylabel("Stock Return")
plt.title("Correlation Scatter Plot")

plt.tight_layout()
plt.show()

##二、回归分析（OLS Regression）
import statsmodels.api as sm
import numpy as np

# 构建回归模型前，清理数据
df_clean = df[['exchange_rate_return','stock_return']].replace([np.inf, -np.inf], np.nan).dropna()

# 构建回归模型：stock_return ~ exchange_rate_return
X = df_clean['exchange_rate_return']
y = df_clean['stock_return']
X = sm.add_constant(X)  # 添加常数项

model = sm.OLS(y, X).fit()
print(model.summary())

##三、换数据频率（Resample）把日度数据转成月度数据，减少噪音，观察长期趋势。

# 按月重采样，计算月度平均汇率变化率和股票收益率
monthly = df.resample('M', on='date').mean()

# 相关性分析
monthly_corr = monthly[['exchange_rate_return','stock_return']].corr()
print("Monthly correlation:\n", monthly_corr)

# 可视化
plt.figure(figsize=(10,5))
plt.scatter(monthly['exchange_rate_return'], monthly['stock_return'])
plt.xlabel("Monthly Exchange Rate Change")
plt.ylabel("Monthly Stock Return")
plt.title("Monthly Correlation Scatter Plot")
plt.show()