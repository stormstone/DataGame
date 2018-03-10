# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 10:26
# @Author  : Storm
# @File    : myPic_predict.py
import datetime

import pandas as pd
import matplotlib.pyplot as plt

data_read = pd.read_table("data/train_20171215.txt", header=0, encoding='gb2312', delim_whitespace=True)

# 统计每天的总数
df_allcnt = data_read.groupby(by=['date'])['cnt'].sum()
df_allcnt = df_allcnt.to_frame()
# 设置索引
df_allcnt['date'] = df_allcnt.index
df_allcnt = df_allcnt.reset_index(drop=True)
plt.show()
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')

filename = 'sub_o03'
data_predict = pd.read_csv('result/' + filename + '.csv')
plt.plot(df_allcnt.date, df_allcnt.cnt, 'bo')
plt.plot(data_predict.iloc[:, 0], data_predict.iloc[:, 1], 'ro')
plt.savefig('result/' + filename + '.png')
plt.show()
plt.plot(df_allcnt.date, df_allcnt.cnt, 'b')
plt.plot(data_predict.iloc[:, 0], data_predict.iloc[:, 1], 'r')
plt.savefig('result/' + filename + '_l.png')

'''
data_predict = pd.read_csv('result/sub_myXgb20180120_170719.csv')
data_predict.iloc[:, 1] = data_predict.iloc[:, 1] + 500
for i in range(len(data_predict)):
    a = data_predict.iloc[i, 1]
    if a > 1000:
        data_predict.iloc[i:i + 1, 1:2] = data_predict.iloc[i:i + 1, 1] + 300
    if a > 4000:
        data_predict.iloc[i:i + 1, 1:2] = data_predict.iloc[i:i + 1, 1] + 1000

submission = pd.DataFrame({'date': data_predict.iloc[:, 0], 'pred': data_predict.iloc[:, 1]})
submission.to_csv(r'result/sub_myXgb20180120_170719addnum.csv', header=None,
                  index=False, float_format='%.0f')
'''
