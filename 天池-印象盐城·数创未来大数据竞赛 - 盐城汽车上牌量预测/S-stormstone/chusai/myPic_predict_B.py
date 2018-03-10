# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 10:26
# @Author  : Storm
# @File    : myPic_predict.py
import datetime

import pandas as pd
import matplotlib.pyplot as plt

data_read = pd.read_table("data/train_20171215.txt", header=0, encoding='gb2312', delim_whitespace=True)
data_testA_result = pd.read_table("data/answer_A_20180225.txt", header=0, encoding='gb2312', delim_whitespace=True)

# 统计每天的总数
df_allcnt = data_read.groupby(by=['date'])['cnt'].sum()
df_allcnt = df_allcnt.to_frame()
# 设置索引
df_allcnt['date'] = df_allcnt.index
df_allcnt = df_allcnt.reset_index(drop=True)

plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')


filename = 'sub_myLgb20180225_202850'
data_predict = pd.read_csv('resultB/' + filename + '.csv')
plt.plot(df_allcnt.date, df_allcnt.cnt, 'bo')
plt.plot(data_testA_result.date, data_testA_result.cnt, 'bo')
plt.plot(data_predict.iloc[:, 0], data_predict.iloc[:, 1], 'ro')
plt.savefig('resultB/' + filename + '.png')
plt.show()
