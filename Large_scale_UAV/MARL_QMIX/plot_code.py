import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

from matplotlib.ticker import FuncFormatter

'''读取csv文件'''


def format_func(value, tick_number):
    if value >= 1e6:
        return f'{value / 1e6:.1f}M'
    elif value >= 1e3:
        return f'{value / 1e3:.0f}k'
    else:
        return f'{value:.0f}'


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    # 读取标题（可选）
    header = next(plots)

    # 找到列的索引
    steps_index = header.index('Step')
    reward_index = header.index('Value')

    # 读取列的数据
    steps_data = [(row[steps_index], row[reward_index]) for row in plots]
    x = []
    y = []
    for index in range(len(steps_data)):
        y.append(float(steps_data[index][1]))
        x.append(float(steps_data[index][0]))
    return x, y


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

plt.figure()
x2, y2 = readcsv("./iql-32.csv")
plt.plot(x2, y2, color='darkorange', label='DQN')
# plt.plot(x2, y2, '.', color='red')

# x, y = readcsv("dloss.csv")
# plt.plot(x, y, 'g', label='Without BN')

x1, y1 = readcsv("./vdn-32new.csv")
plt.plot(x1, y1, color='yellow', label='VDN')
# plt.plot(x1, y1, color='black', label='Without DW and PW')
x3, y3 = readcsv("./qmix-32new.csv")
# plt.plot(x4, y4, color='mediumblue', label='QMIX', linestyle='dashed')
plt.plot(x3, y3, color='black', label='QMIX-poly')


x4, y4 = readcsv("./qmix-32.csv")
# plt.plot(x4, y4, color='mediumblue', label='QMIX', linestyle='dashed')

plt.plot(x4, y4, color='mediumblue', label='QMIX')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(-300, 130)
plt.xlim(0, 2000000)

# 设置横纵坐标的名称以及对应字体格式
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=20)  # matplotlib内无中文字节码，需要自行手动添加
plt.xlabel('train_steps', font2, fontproperties=font_set)
plt.ylabel('Reward', font2, fontproperties=font_set)
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
plt.legend(fontsize=16, prop=font1)

# 设置汉字显示正确
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# 设置保存完全的图
plt.tight_layout()
plt.savefig('reward-32.png')
plt.show()
