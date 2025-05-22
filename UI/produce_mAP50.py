import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df1 = pd.read_csv("results1.csv")  # 读取文件 1
df2 = pd.read_csv("results2.csv")  # 读取文件 2

epoch_1 = df1["               epoch"].values.tolist()  # 通过文件表头信息读取文件内容
mAP5_1 = df1["     metrics/mAP_0.5"].values.tolist()

epoch_2 = df2["               epoch"].values.tolist()  # 通过文件表头信息读取文件内容
mAP5_2 = df2["     metrics/mAP_0.5"].values.tolist()

plt.figure(figsize=(8, 5))
plt.plot(epoch_1, mAP5_1, color='red', label='yolov5s 改进算法')  # 设置曲线相关系数
plt.plot(epoch_2, mAP5_2, color='black', label='yolov5s 原始算法')  # 设置曲线相关系数

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.ylim(0, 1)
plt.xlim(0, 100)  # 设置坐标轴取值范围
plt.xlabel('epochs', fontsize=14)
plt.ylabel('mAP_0.5', fontsize=14)
plt.legend(fontsize=12, loc="best")  # 设置标签位置及大小

# 修改保存路径
plt.savefig(r"E:\desk\biyesheji\duplication1\picture\test.png", bbox_inches='tight')
plt.show()
