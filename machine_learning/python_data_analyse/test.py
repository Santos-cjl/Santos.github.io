import numpy as np
import pandas as pd

#第一部分：numpy的应用

name = np.array([100,121,212])
                 #[770,666,0]])
print(name)
print(type(name))
print(name.dtype)
print(name.shape)
print(name.ndim)
#print(name.T)
print(name.argmax())

#综合题 1：函数版组合表现分析
# 给定 price_matrix 和 weights，所有人使用同一组输入数据，编写 compute_returns(price_matrix) 函数计算收益率矩阵。
# 编写 build_portfolio(price_matrix, weights) 函数，计算组合日收益率，计算累计净值并找出最好交易日和最好交易日收益率并且打印。
import numpy as np

price_matrix = np.array([
    [100.0, 120.0, 80.0],
    [102.0, 118.0, 82.0],
    [101.0, 121.0, 81.0],
    [104.0, 123.0, 83.0],
    [103.0, 122.0, 84.0]
])
weights = np.array([0.5, 0.3, 0.2])

def compute_returns(price_matrix):
    return price_matrix[1:] / price_matrix[:-1] - 1

def build_portfolio(price_matrix, weights):
    returns = compute_returns(price_matrix)
    portfolio_ret = returns @ weights
    nav = np.cumprod(1 + portfolio_ret)
    best_day = portfolio_ret.argmax()
    return (portfolio_ret, nav, best_day, portfolio_ret[best_day])

results = build_portfolio(price_matrix, weights)

print("组合日收益率为：{}\n累计净值为：{}\n最好交易日为第{}天\n最差交易收益率为：{}".format(results[0],results[1],results[2],results[3]))

#综合题 2：循环版组合风险分析
#已给定下面的 returns 和 weights，所有人使用同一组输入数据。
#请使用循环逐列计算各资产平均收益，并用循环识别压力交易日(存在资产绝对值收益率大于2%)。
#使用 NumPy计算相关系数、协方差、组合方差和组合波动率。
#提示：组合波动率为组合资产收益率标准差，用 np.sqrt(variance) 计算。
import numpy as np

returns = np.array([
    [0.012, 0.005, -0.002],
    [-0.004, 0.011, 0.003],
    [0.021, -0.018, 0.006],
    [0.007, -0.002, 0.026],
    [-0.013, 0.004, -0.001]
])
weights = np.array([0.5, 0.3, 0.2])

mean_list = []
for col in range(returns.shape[1]):
    mean_list.append(returns[:, col].mean())
mean_vector = np.array(mean_list)

stress_days = []
for row in range(returns.shape[0]):
    if np.any(np.abs(returns[row]) > 0.02):
        stress_days.append(row)

corr_matrix = np.corrcoef(returns, rowvar=False)
cov_matrix = np.cov(returns, rowvar=False)
portfolio_var = weights.T @ cov_matrix @ weights
portfolio_vol = np.sqrt(portfolio_var)

print(mean_vector)
print(corr_matrix)
print(cov_matrix)
print(portfolio_var)
print(portfolio_vol)
print(stress_days)

#第二部分：pandas的应用

#DataFrame的创建
import pandas as pd

# df = pd.DataFrame({
   # "date": ["2026/4/1", "2026/4/1", "2026/4/1"],
   # "code": ["000001.SZ", "600036.SH", "601318.SH"],
   # "close": [11.15, 45.82, 53.71]
#})

#print(df)

#读取文件
import pandas as pd

df = pd.read_csv("data.csv")
# df = pd.read_excel("data.xlsx")

print(df.head())
print(df.shape)
print(df.columns.tolist())

#快速看表
print(df.shape)
print(len(df))
print(df.columns.tolist())
print(df.dtypes)
print(df.head())
print(df.tail())
print(df["code"].nunique())
print(df["date"].nunique())
print(df.describe())
print(df.isna().sum())


#综合题 3：data.csv 多股票组合表现分析
#输入asset_list，将数据集中的相应股票筛选出来并且添加上股票对应的行业
#计算asset_list中的个股收益率
#计算该投资策略的加权收益以及累计收益
#计算不同行业的回报率均值、方差以及最大值
#展现该投资组合的周度收益

import numpy as np
import pandas as pd

asset_list = ["000001.SZ", "600036.SH", "601318.SH", "600519.SH"]
weight_map = {
    "000001.SZ": 0.25,
    "600036.SH": 0.30,
    "601318.SH": 0.20,
    "600519.SH": 0.25,
}
sector_map = {
    "000001.SZ": "Bank",
    "600036.SH": "Bank",
    "601318.SH": "Insurance",
    "600519.SH": "Consumer",
}
focus_assets = set(asset_list)

def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def compute_returns(df, asset_list):
    out = df[df["code"].isin(asset_list)].sort_values(["code", "date"]).copy()
    out["daily_return"] = out.groupby("code")["close"].pct_change()
    return out

def build_portfolio(ret_panel, asset_list, weight_map):
    wide = ret_panel.pivot(index="date", columns="code", values="daily_return").sort_index()
    weight = np.array([weight_map[a] for a in asset_list])
    port_ret = wide[asset_list].fillna(0).to_numpy() @ weight
    ans = pd.DataFrame({"date": wide.index, "portfolio_return": port_ret})
    ans["nav"] = np.cumprod(1 + ans["portfolio_return"])
    return ans

def summarize_by_sector(ret_panel, sector_map):
    out = ret_panel.dropna(subset=["daily_return"]).copy()
    out["sector"] = out["code"].map(sector_map)
    return out.groupby("sector")["daily_return"].agg(["mean", "std", "max"])

df = load_data("data.csv")
ret_panel = compute_returns(df, asset_list)
portfolio = build_portfolio(ret_panel, asset_list, weight_map)
weekly = portfolio.set_index("date")["portfolio_return"].resample("W").mean()
sector_summary = summarize_by_sector(ret_panel, sector_map)
print(portfolio.head())
print(weekly)
print(sector_summary)

#综合题 4：因子分组与风险状态分析
#读取 data.csv，将 date 转为日期类型，并仅保留 code 属于 asset_list 的样本，根据 sector_map 为样本添加行业列 sector
#构造 intraday_return、amplitude、volume_rank、return_rank、factor_score，amplitude = (high - low) / close，factor_score = return_rank + 0.3 * volume_rank - 0.2 * amplitude
#按 date 分组，先对 factor_score 做 rank(method="first")
#再用 qcut 将样本划分为 5 组，生成 factor_group
#按 sector 统计各行业的 factor_score 均值

import numpy as np
import pandas as pd

asset_list = {"000001.SZ", "600036.SH", "601318.SH", "600519.SH", "300750.SZ"}
sector_map = {
    "000001.SZ": "Bank",
    "600036.SH": "Bank",
    "601318.SH": "Insurance",
    "600519.SH": "Consumer",
    "300750.SZ": "Manufacturing",
}

import pandas as pd
asset_list = ["000001.SZ", "600036.SH", "601318.SH", "600519.SH", "300750.SZ"]
sector_map = {
    "000001.SZ": "Bank",
    "600036.SH": "Bank",
    "601318.SH": "Insurance",
    "600519.SH": "Consumer",
    "300750.SZ": "Manufacturing",
}

def prepare_factor_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["code"].isin(asset_list)].copy()
    df["sector"] = df["code"].map(sector_map)
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
    df["amplitude"] = (df["high"] - df["low"]) / df["close"]
    df["volume_rank"] = df.groupby("date")["volume"].rank(pct=True)
    df["return_rank"] = df.groupby("date")["intraday_return"].rank(pct=True)
    df["factor_score"] = (
        df["return_rank"]
        + 0.3 * df["volume_rank"]
        - 0.2 * df["amplitude"]
    )
    return df
def assign_factor_group(df):
    out = df.copy()

    out["factor_group"] = out.groupby("date")["factor_score"].transform(
        lambda x: pd.qcut(
            x.rank(method="first"),
            5,
            labels=False,
            duplicates="drop"
        )
    )

    return out
def summarize_sector(panel):
    sector_mean = (
        panel.groupby("sector", dropna=False)["factor_score"]
        .mean()
        .reset_index(name="avg_factor_score")
        .sort_values("avg_factor_score", ascending=False)
    )
    return sector_mean
panel = assign_factor_group(prepare_factor_data("data.csv"))
sector_mean = summarize_sector(panel)
print(panel.head())
print(sector_mean)

##  matplotlib的应用

import matplotlib.pyplot as plt
#项目 3：组合表现仪表盘
#基于 data.csv 和给定权重，生成组合净值、回撤和成交量图。要求复用前面计算组合收益率的思路。
asset_list = ["000001.SZ", "600036.SH", "601318.SH"]
weight_map = {"000001.SZ": 0.3, "600036.SH": 0.4, "601318.SH": 0.3}

# 任务：
# 1. 读取 data.csv 并计算收益率宽表。
# 2. 根据权重得到组合收益率与净值。
# 3. 画出三行子图：组合净值、组合回撤、三只股票成交量。
# 4. 设置标题、图例、日期刻度，并保存为 portfolio_dashboard.png。

asset_list = ["000001.SZ", "600036.SH", "601318.SH"]
weight_map = {"000001.SZ": 0.3, "600036.SH": 0.4, "601318.SH": 0.3}

plot_df = df.copy()
plot_df["date"] = pd.to_datetime(plot_df["date"])
plot_df = plot_df[plot_df["code"].isin(asset_list)].sort_values(["code", "date"]).copy()
plot_df["daily_return"] = plot_df.groupby("code")["close"].pct_change().fillna(0)

ret_wide = plot_df.pivot(index="date", columns="code", values="daily_return").sort_index()
vol_wide = plot_df.pivot(index="date", columns="code", values="volume").sort_index()
weights = np.array([weight_map[a] for a in asset_list])
portfolio_ret = pd.Series(ret_wide[asset_list].to_numpy() @ weights, index=ret_wide.index)
nav = (1 + portfolio_ret).cumprod()
drawdown = nav / nav.cummax() - 1

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(nav.index, nav.values, label="NAV", color="#00A896")
axes[1].fill_between(drawdown.index, drawdown.values, 0, color="#F25F5C", alpha=0.35)
vol_wide[asset_list].plot(ax=axes[2])
axes[0].set_title("Portfolio Dashboard")
axes[0].legend()
axes[2].legend(title="Code")
axes[2].tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig("portfolio_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

### 课程总结
# NumPy 负责高效的矩阵与向量计算，适合收益率、权重、协方差和组合风险。
# Pandas 负责表格、分组、透视、合并和时间序列，是金融面板数据分析的主力工具。
# Matplotlib 负责进行结果展示。
# 专业的数据分析包含数据清理、数据计算和数据结果展示。
