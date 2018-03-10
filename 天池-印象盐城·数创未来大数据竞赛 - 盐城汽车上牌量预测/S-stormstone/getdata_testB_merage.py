# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 13:39
# @Author  : 石头人m
# @File    : getdata_testB_merage.py

import pandas as pd

testB_1351 = pd.read_csv('getdata/getdata_testB_1351.csv')
testB_1352 = pd.read_csv('getdata/getdata_testB_1352.csv')

testB = pd.concat([testB_1351, testB_1352])

testB.to_csv('getdata/getdata_testB.csv', index=None)
