# -*- coding: utf-8 -*-
# @Time    : 2018-01-09 18:56
# @Author  : Storm
# @File    : dataprocess.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data_test_read = pd.read_table("data/test_A_20171225.txt", header=-1, encoding='gb2312', delim_whitespace=True)
data_read = pd.read_table("data/train_20171215.txt", header=0, encoding='gb2312', delim_whitespace=True)

# 统计每天的总数
df_allcnt = data_read.groupby(by=['date'])['cnt'].sum()
df_allcnt = df_allcnt.to_frame()

# 设置索引
df_allcnt['date'] = df_allcnt.index
df_allcnt = df_allcnt.reset_index(drop=True)

# 画图
plt.plot(df_allcnt.date, df_allcnt.cnt, 'bo', label="point")
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')
plt.savefig("pic/总量-时间-趋势.jpg")
plt.show()

# 根据日期，星期分组
df_train = data_read.groupby(by=['date', 'day_of_week'])['cnt'].sum()
df_train = df_train.to_frame()
df_train.to_csv('result/dataP_train01_groupByDD.csv')

# =======================================================================================
# 添加周期余数、取模
df_groupByDD = pd.read_csv('result/dataP_train01_groupByDD.csv')
# df_groupByDD['cycle1'] = df_groupByDD['date'] % 310
df_groupByDD['cycle2'] = df_groupByDD['date'] % 320
# df_groupByDD['cycle3'] = df_groupByDD['date'] % 330
# df_groupByDD['cycle4'] = df_groupByDD['date'] % 340
df_groupByDD['cycle5'] = df_groupByDD['date'] % 350
df_groupByDD['cycle5_2'] = df_groupByDD['date'] % 354
# df_groupByDD['cycle5_3'] = df_groupByDD['date'] % 365

# df_groupByDD['cycle6'] = df_groupByDD['date'] / 310
# df_groupByDD['cycle7'] = df_groupByDD['date'] / 320
# df_groupByDD['cycle8'] = df_groupByDD['date'] / 330
# df_groupByDD['cycle9'] = df_groupByDD['date'] / 340
df_groupByDD['cycle10'] = df_groupByDD['date'] / 350
df_groupByDD.to_csv('result/dataP_train02_cycle.csv', index=None)
# df_groupByDD.to_csv('result/dataP_train02_cycle02.csv', index=None)
# =======================================================================================

# 处理测试数据
# 添加周期余数、取模
data_test_read.to_csv('result/dataP_test01_read.csv', index=None, header=0)
df_test = pd.read_csv('result/dataP_test01_read.csv')
# df_test['cycle1'] = df_test['date'] % 310
df_test['cycle2'] = df_test['date'] % 320
# df_test['cycle3'] = df_test['date'] % 330
# df_test['cycle4'] = df_test['date'] % 340
df_test['cycle5'] = df_test['date'] % 350
df_test['cycle5_2'] = df_test['date'] % 354
# df_test['cycle5_3'] = df_test['date'] % 365

# df_test['cycle6'] = df_test['date'] / 310
# df_test['cycle7'] = df_test['date'] / 320
# df_test['cycle8'] = df_test['date'] / 330
# df_test['cycle9'] = df_test['date'] / 340
df_test['cycle10'] = df_test['date'] / 350
df_test.to_csv('result/dataP_test02_cycle.csv', index=None)
# df_test.to_csv('result/dataP_test02_cycle02.csv', index=None)
# =======================================================================================

# PolynomialFeatures
def polyX(df_train_X, df_test):
    poly = PolynomialFeatures(2, interaction_only=False)  # 默认的阶数是２，同时设置交互关系为true
    X = pd.concat([df_train_X, df_test])
    polyX = poly.fit_transform(X)
    return polyX


def polyX_go():
    predictors = [f for f in df_test.columns if f not in ['cnt']]
    df_train_X = df_groupByDD[predictors]
    my_polyX = polyX(df_train_X, df_test)
    pd.DataFrame(my_polyX).to_csv('result/dataP_my_polyX02.csv', index=None)

    # 读取my_polyX
    my_polyX = pd.read_csv('result/dataP_my_polyX02.csv')
    polyX_train = my_polyX[:len(df_train_X)]
    polyX_test = my_polyX[len(df_train_X):]

    df_test_poly = polyX_test
    df_test_poly['date'] = pd.Series(df_test['date'].values, index=df_test_poly.index)
    df_test_poly['day_of_week'] = pd.Series(df_test['day_of_week'].values, index=df_test_poly.index)

    df_train_poly = polyX_train
    df_train_poly['cnt'] = df_groupByDD['cnt']
    df_train_poly['date'] = df_groupByDD['date']
    df_train_poly['day_of_week'] = df_groupByDD['day_of_week']

    df_train_poly.to_csv('result/dataP_train03_poly02.csv', index=None)
    df_test_poly.to_csv('result/dataP_test03_poly02.csv', index=None)

polyX_go()