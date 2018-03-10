# -*- coding: utf-8 -*-
# @Time    : 2018/2/27 22:06
# @Author  : 石头人m
# @File    : getdata_testAB

import pandas as pd
import numpy as np
import datetime

testB = pd.read_table('data_fusai/fusai_test_B_20180227.txt')

testB_1352 = testB[testB['date'] > 1351]
train = testB_1352
nowtime = datetime.date(2017, 1, 14)
filename = 'getdata_testB_1352'

print(train.head())

# print(starttime+datetime.timedelta(days=32))
timelist = []
l = np.shape(train)[0] - 1
print(l)
i = 926
l = l + 926
while i < l:
    d = [nowtime, train.loc[i, 'date']]
    timelist.append(d)
    while i < l and train.loc[i, 'day_of_week'] <= train.loc[i + 1, 'day_of_week']:
        if train.loc[i, 'day_of_week'] < train.loc[i + 1, 'day_of_week']:
            nowtime = nowtime + datetime.timedelta(days=1)
            d = [nowtime, train.loc[i + 1, 'date']]
            timelist.append(d)
        i = i + 1
    md = train.loc[i, 'day_of_week']
    while md < 7:
        nowtime = nowtime + datetime.timedelta(days=1)
        d = [nowtime, np.nan]
        timelist.append(d)
        md = md + 1
    i = i + 1
    if i >= l:
        break
    md = train.loc[i, 'day_of_week']
    k = 1
    while k < md:
        nowtime = nowtime + datetime.timedelta(days=1)
        d = [nowtime, np.nan]
        timelist.append(d)
        k = k + 1
    nowtime = nowtime + datetime.timedelta(days=1)
timelist = timelist[:-3]
# print(timelist)
r_timelist = []
for t in timelist:
    for br in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        temp = t.copy()
        temp.append(br)
        r_timelist.append(temp)
data = pd.DataFrame(r_timelist)
data.columns = ['r_date', 'date', 'brand']

train = pd.merge(data, train, how='left', on=['date', 'brand'])
data = pd.merge(data, train, how='left', on=['r_date', 'brand', 'date'])

data.to_csv('getdata/' + filename + '.csv', index=None)
'''
'''
