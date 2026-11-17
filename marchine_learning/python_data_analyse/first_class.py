# NO1.输入日期并分析天数
import datetime

dtstr = input('Please input the datetime:')
dt = datetime.datetime.strptime(dtstr, "%Y%m%d")
another_dtstr = dtstr[:4] + '0101'
another_dt = datetime.datetime.strptime(another_dtstr, "%Y%m%d")
print(int((dt - another_dt).days) + 1)

# NO2.猜数字游戏
import random

secret = random.randint(1, 100)
print("please guess a number, and you can guess less than six times")
tries = 1

while tries <= 6:
    guess = int(input(f'1-100 use 3d time please input a number: % (tries,)'))
    if guess == secret:
        print("you are right!")
        break
    elif guess > secret:
        print("the number is bigger")
    else:
        print("the number is smaller")
    tries += 1
else:
    print("see you")

# No3.输入一行字符，分别统计出其中中英文字母、空格、数字和其他字符的个数
import string

s = input("please input the string:")
letter = 0
space = 0
digit = 0
other = 0
for c in s:
    if c.isalpha():
        letter += 1
    elif c.isdigit():
        digit += 1
    elif c.isspace():
        space += 1
    else:
        other += 1
print('There are %d letters,%d spaces and %d digits' % (letter, space, digit))

