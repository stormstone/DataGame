# -*- coding: utf-8 -*-
# @Time    : 2018-01-18 18:00
# @Author  : Storm
# @File    : other01.py

import pandas as pd
import matplotlib.pyplot as plt

dir = '../data/'
train = pd.read_table(dir + 'train_20171215.txt', engine='python')
test_A = pd.read_table(dir + 'test_A_20171225.txt', engine='python')
sample_A = pd.read_table(dir + 'sample_A_20171225.txt', engine='python', header=None)

print(train.info())
print(test_A.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4773 entries, 0 to 4772
Data columns (total 4 columns):
date           4773 non-null int64
day_of_week    4773 non-null int64
brand          4773 non-null int64
cnt            4773 non-null int64
dtypes: int64(4)
memory usage: 149.2 KB
None
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 276 entries, 0 to 275
Data columns (total 2 columns):
date           276 non-null int64
day_of_week    276 non-null int64
dtypes: int64(2)
memory usage: 4.4 KB
None
'''

print(train['day_of_week'].unique())
print(test_A['day_of_week'].unique())
# [3 4 5 6 7 1 2]
# [4 5 6 1 2 3 7]

# cnt箱型图
plt.boxplot(train['cnt'])
plt.savefig('../pic/cnt-箱型图.png')
plt.show()

import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew

# cnt分布
sns.distplot(train['cnt'], fit=norm)
plt.plot(train['date'], train['cnt'])
plt.savefig('../pic/cnt-分布图.png')
plt.show()
print(train['cnt'].describe())
'''
count    4773.000000
mean      380.567358
std       252.720918
min        12.000000
25%       221.000000
50%       351.000000
75%       496.000000
max      2102.000000
Name: cnt, dtype: float64
'''

from sklearn.metrics import mean_squared_error

train['25%'] = 221
train['50%'] = 351
train['75%'] = 496
train['median'] = train['cnt'].median()
train['mean'] = train['cnt'].mean()
print(mean_squared_error(train['cnt'], train['25%']))
print(mean_squared_error(train['cnt'], train['50%']))
print(mean_squared_error(train['cnt'], train['75%']))
print(mean_squared_error(train['cnt'], train['median']))
print(mean_squared_error(train['cnt'], train['mean']))
'''
89316.2231301
64728.7100356
77179.1761995
64728.7100356
63854.4813732
'''

# 星期几的总量的趋势
for i in range(7):
    monday = train[train['day_of_week'] == i + 1]
    plt.plot(range(len(monday)), monday['cnt'])
    plt.title('总量-星期%d' % (i + 1))
    plt.savefig('../pic/总量-星期%d.png' % (i + 1))
    plt.show()

res = train.groupby(['day_of_week'], as_index=False).cnt.mean()
xx = train.merge(res, on=['day_of_week'])
print(xx.head())
print(mean_squared_error(xx['cnt_x'], xx['cnt_y']))
'''                                                                         
 date  day_of_week  brand  cnt_x  25%  50%  75%  median        mean          cnt_y  
0     1            3      1     20  221  351  496   351.0  380.567358   425.141451  
1     1            3      5     48  221  351  496   351.0  380.567358   425.141451  
2     8            3      1    569  221  351  496   351.0  380.567358   425.141451  
3     8            3      2    532  221  351  496   351.0  380.567358   425.141451  
4     8            3      3    674  221  351  496   351.0  380.567358   425.141451  

48035.0908982
'''
