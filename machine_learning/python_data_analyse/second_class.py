# No.1
import math

# 1. 获取用户输入并转换为浮点数
try:
    num = float(input("请输入一个数字："))
except ValueError:
    print("输入错误，请输入有效的数字！")
    exit()

# 2. 多分支条件判断与运算
if num < 10:
    # 大于10，计算sin值（math.sin默认弧度，如需角度可先转弧度：math.radians(num)）
    result = math.sin(num)
    print(f"数字 {num} 小于10，sin值为：{result:.4f}")
elif 10 <= num <= 20:
    # 10到20之间，计算cos值
    result = math.cos(num)
    print(f"数字 {num} 在10-20之间，cos值为：{result:.4f}")
elif 20 < num < 30:
    # 20到30之间，执行加法运算（示例：数字+10，可修改为其他加法逻辑）
    add_num = 10  # 自定义加数
    result = num + add_num
    print(f"数字 {num} 在20-30之间，加法运算（+{add_num}）结果为：{result}")
else:
    print(f"数字 {num} 不在指定运算区间内！")

# No.2
import math

def is_two_decimal(num):
    """判断数字是否为两位小数（兼容整数、浮点数输入）"""
    # 转字符串后按小数点分割，判断小数部分长度
    str_num = str(num).rstrip('0').rstrip('.')  # 去除末尾0和多余小数点
    if '.' in str_num:
        integer_part, decimal_part = str_num.split('.')
        return len(decimal_part) == 2
    return False  # 整数不是两位小数

# 1. 获取用户输入
try:
    num = float(input("请输入一个数字："))
except ValueError:
    print("输入错误，请输入有效的数字！")
    exit()

# 2. 多分支条件判断（注意：先判断特殊条件，再判断区间）
if num > 30 and is_two_decimal(num):
    # 大于30且为两位小数，执行减法运算（示例：数字-5，可自定义）
    sub_num = 5  # 自定义减数
    result = num - sub_num
    print(f"数字 {num} 大于30且为两位小数，减法运算（-{sub_num}）结果为：{result}")
elif num < 10:
    result = math.sin(num)
    print(f"数字 {num} 小于10，sin值为：{result:.4f}")
elif 10 <= num <= 20:
    result = math.cos(num)
    print(f"数字 {num} 在10-20之间，cos值为：{result:.4f}")
elif 20 < num < 30:
    add_num = 10
    result = num + add_num
    print(f"数字 {num} 在20-30之间，加法运算（+{add_num}）结果为：{result}")
else:
    print(f"数字 {num} 不在指定运算区间内！")

# No.3
def bmi_classify(bmi):
    if bmi < 18.5:
        return "太瘦"
    elif 18.5 <= bmi < 20:
        return "偏瘦"
    elif 20 <= bmi < 24:
        return "中等"
    elif 24 <= bmi < 28:
        return "偏胖"
    else:
        return "很胖"

# 1. 获取用户输入（身高：米，体重：千克）
try:
    height = float(input("请输入你的身高（单位：米，如1.75）："))
    weight = float(input("请输入你的体重（单位：千克，如65）："))
except ValueError:
    print("输入错误，请输入有效的数字！")
    exit()

# 2. 计算BMI
bmi = weight / (height ** 2)
# 3. 获取分级结果
result = bmi_classify(bmi)

# 4. 输出结果
print(f"\n你的BMI值为：{bmi:.2f}")
print(f"身体指数分级结果：{result}")