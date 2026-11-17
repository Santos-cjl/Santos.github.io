import numpy as np

# 1. 给定题目里的价格矩阵和权重
price_matrix = np.array([
    [100.0, 120.0, 80.0],
    [102.0, 118.0, 82.0],
    [105.0, 121.0, 81.0],
    [104.0, 123.0, 84.0],
    [103.0, 122.0, 84.0]
])

weights = np.array([0.5, 0.3, 0.2])  # 三只资产的配置权重


# 2. 定义计算收益率矩阵的函数
def compute_returns(price_matrix):
    """
    输入: 价格矩阵 (行=交易日, 列=不同资产)
    输出: 每日收益率矩阵
    """
    # t日价格 / t-1日价格 - 1
    returns_matrix = price_matrix[1:] / price_matrix[:-1] - 1
    return returns_matrix


# 3. 定义构建投资组合、分析表现的函数
def build_portfolio(price_matrix, weights):
    # 先拿到收益率矩阵
    ret_mat = compute_returns(price_matrix)

    # 计算每日投资组合收益率 (矩阵乘法: 每行收益率 × 权重)
    daily_portfolio_returns = ret_mat @ weights

    # 计算累计净值 (初始净值=1)
    cumulative_nav = np.cumprod(1 + daily_portfolio_returns)

    # 找到收益率最高、最低的交易日索引和数值
    best_day_idx = np.argmax(daily_portfolio_returns)
    worst_day_idx = np.argmin(daily_portfolio_returns)

    best_return = daily_portfolio_returns[best_day_idx]
    worst_return = daily_portfolio_returns[worst_day_idx]

    # 打印结果
    print("==== 收益率矩阵 ====")
    print(np.round(ret_mat, 6))
    print("\n==== 组合每日收益率 ====")
    print(np.round(daily_portfolio_returns, 6))
    print("\n==== 累计净值走势 ====")
    print(np.round(cumulative_nav, 6))
    print(f"\n✅ 最佳交易日(索引从0开始): 第{best_day_idx + 1}个组合交易日, 收益率: {best_return:.4%}")
    print(f"❌ 最差交易日(索引从0开始): 第{worst_day_idx + 1}个组合交易日, 收益率: {worst_return:.4%}")

    return daily_portfolio_returns, cumulative_nav, best_day_idx, worst_day_idx


# 4. 运行函数
build_portfolio(price_matrix, weights)