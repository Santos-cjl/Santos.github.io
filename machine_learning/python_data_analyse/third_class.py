# NO.1写一个计算BMI的函数，函数名为cal_BMI
# 定义cal_BMI函数
def cal_BMI(weight, height):
    # 1. 计算BMI数值
    bmi = weight / (height ** 2)

    # 2. 判断BMI分类
    if bmi < 20:
        category = "偏瘦"
    elif 20 <= bmi < 25:
        category = "正常"
    elif 25 <= bmi < 30:
        category = "偏胖"
    # 题目写35以上为太胖，这里补充30-35区间
    elif 30 <= bmi <=35:
        category = "重度偏胖"
    else:  # 35以上
        category = "太胖"

    # 3. 打印分类结果
    print(f"你的BMI分类为：{category}")

    # 4. 返回BMI计算值
    return bmi


# 测试调用
if __name__ == "__main__":
    # 示例：体重70kg，身高1.75m
    result = cal_BMI(70, 1.75)
    print(f"本次计算的BMI数值为：{result:.2f}")