# -*- coding: utf-8 -*-
# @Time    : 18-2-27 下午11:22
# @Author  : Storm
# @File    : getdata_pic.py

import pandas as pd
import matplotlib.pyplot as plt

data_path = 'getdata/'
train = pd.read_csv(data_path + 'getdata_train_ok.csv')

for i in range(10):
    train_brand = train.loc[train['brand'] == i + 1]
    plt.title('brand-%d'% (i + 1))
    plt.xlabel('Date')
    plt.ylabel('Cnt')
    plt.plot(train_brand.date, train_brand.cnt, 'bo')
    plt.savefig('pic_brand/brand{}.png'.format(i+1))
    plt.show()
    plt.title('brand-%d'% (i + 1))
    plt.xlabel('Date')
    plt.ylabel('Cnt')
    plt.plot(train_brand.date, train_brand.cnt, 'b')
    plt.savefig('pic_brand/brand{}_l.png'.format(i+1))
    plt.show()
