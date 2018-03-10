# -*- coding: utf-8 -*-
# @Time    : 2018-01-20 15:36
# @Author  : Storm
# @File    : myPic.py

import pandas as pd
import matplotlib.pyplot as plt

data_test_read = pd.read_table("data/test_A_20171225.txt", header=-1, encoding='gb2312', delim_whitespace=True)
data_read = pd.read_table("data/train_20171215.txt", header=0, encoding='gb2312', delim_whitespace=True)

# 统计每天的总数
df_allcnt = data_read.groupby(by=['date'])['cnt'].sum()
df_allcnt = df_allcnt.to_frame()
# 设置索引
df_allcnt['date'] = df_allcnt.index
df_allcnt = df_allcnt.reset_index(drop=True)
plt.show()
plt.plot(df_allcnt.date, df_allcnt.cnt, 'b', label="point")
plt.title('All Count Law')
plt.xlabel('Date')
plt.ylabel('Cnt')
plt.savefig("pic/总量-时间-趋势.jpg")
plt.show()

# 根据日期，品牌分组
df_train_brand1 = data_read[data_read['brand'] == 1]
df_train_brand2 = data_read[data_read['brand'] == 2]
df_train_brand3 = data_read[data_read['brand'] == 3]
df_train_brand4 = data_read[data_read['brand'] == 4]
df_train_brand5 = data_read[data_read['brand'] == 5]
plt.plot(df_train_brand1.date, df_train_brand1.cnt, 'b', label="point")
plt.savefig('pic/品牌1-总量-趋势.png')
plt.show()
plt.plot(df_train_brand2.date, df_train_brand2.cnt, 'b', label="point")
plt.savefig('pic/品牌2-总量-趋势.png')
plt.show()
plt.plot(df_train_brand3.date, df_train_brand3.cnt, 'b', label="point")
plt.savefig('pic/品牌3-总量-趋势.png')
plt.show()
plt.plot(df_train_brand4.date, df_train_brand4.cnt, 'b', label="point")
plt.savefig('pic/品牌4-总量-趋势.png')
plt.show()
plt.plot(df_train_brand5.date, df_train_brand5.cnt, 'b', label="point")
plt.savefig('pic/品牌5-总量-趋势.png')
plt.show()

# 每个品牌每个星期几变化
df_train_brand1_week1 = df_train_brand1[df_train_brand1['day_of_week'] == 1]
df_train_brand1_week2 = df_train_brand1[df_train_brand1['day_of_week'] == 2]
df_train_brand1_week3 = df_train_brand1[df_train_brand1['day_of_week'] == 3]
df_train_brand1_week4 = df_train_brand1[df_train_brand1['day_of_week'] == 4]
df_train_brand1_week5 = df_train_brand1[df_train_brand1['day_of_week'] == 5]
df_train_brand1_week6 = df_train_brand1[df_train_brand1['day_of_week'] == 6]
df_train_brand1_week7 = df_train_brand1[df_train_brand1['day_of_week'] == 7]
plt.plot(df_train_brand1_week1.date, df_train_brand1_week1.cnt, 'b', label="point")
plt.savefig('pic/品牌1-星期1-趋势.png')
plt.show()
plt.plot(df_train_brand1_week2.date, df_train_brand1_week2.cnt, 'b', label="point")
plt.savefig('pic/品牌1-星期2-趋势.png')
plt.show()
plt.plot(df_train_brand1_week3.date, df_train_brand1_week3.cnt, 'b', label="point")
plt.savefig('pic/品牌1-星期3-趋势.png')
plt.show()
plt.plot(df_train_brand1_week4.date, df_train_brand1_week4.cnt, 'b', label="point")
plt.savefig('pic/品牌1-星期4-趋势.png')
plt.show()
plt.plot(df_train_brand1_week5.date, df_train_brand1_week5.cnt, 'b', label="point")
plt.savefig('pic/品牌1-星期5-趋势.png')
plt.show()
plt.plot(df_train_brand1_week6.date, df_train_brand1_week6.cnt, 'b', label="point")
plt.savefig('pic/品牌1-星期6-趋势.png')
plt.show()
plt.plot(df_train_brand1_week7.date, df_train_brand1_week7.cnt, 'b', label="point")
plt.savefig('pic/品牌1-星期7-趋势.png')
plt.show()

df_train_brand2_week1 = df_train_brand2[df_train_brand2['day_of_week'] == 1]
df_train_brand2_week2 = df_train_brand2[df_train_brand2['day_of_week'] == 2]
df_train_brand2_week3 = df_train_brand2[df_train_brand2['day_of_week'] == 3]
df_train_brand2_week4 = df_train_brand2[df_train_brand2['day_of_week'] == 4]
df_train_brand2_week5 = df_train_brand2[df_train_brand2['day_of_week'] == 5]
df_train_brand2_week6 = df_train_brand2[df_train_brand2['day_of_week'] == 6]
df_train_brand2_week7 = df_train_brand2[df_train_brand2['day_of_week'] == 7]
plt.show()
plt.plot(df_train_brand2_week1.date, df_train_brand2_week1.cnt, 'b', label="point")
plt.savefig('pic/品牌2-星期1-趋势.png')
plt.show()
plt.plot(df_train_brand2_week2.date, df_train_brand2_week2.cnt, 'b', label="point")
plt.savefig('pic/品牌2-星期2-趋势.png')
plt.show()
plt.plot(df_train_brand2_week3.date, df_train_brand2_week3.cnt, 'b', label="point")
plt.savefig('pic/品牌2-星期3-趋势.png')
plt.show()
plt.plot(df_train_brand2_week4.date, df_train_brand2_week4.cnt, 'b', label="point")
plt.savefig('pic/品牌2-星期4-趋势.png')
plt.show()
plt.plot(df_train_brand2_week5.date, df_train_brand2_week5.cnt, 'b', label="point")
plt.savefig('pic/品牌2-星期5-趋势.png')
plt.show()
plt.plot(df_train_brand2_week6.date, df_train_brand2_week6.cnt, 'b', label="point")
plt.savefig('pic/品牌2-星期6-趋势.png')
plt.show()
plt.plot(df_train_brand2_week7.date, df_train_brand2_week7.cnt, 'b', label="point")
plt.savefig('pic/品牌2-星期7-趋势.png')

df_train_brand3_week1 = df_train_brand3[df_train_brand3['day_of_week'] == 1]
df_train_brand3_week2 = df_train_brand3[df_train_brand3['day_of_week'] == 2]
df_train_brand3_week3 = df_train_brand3[df_train_brand3['day_of_week'] == 3]
df_train_brand3_week4 = df_train_brand3[df_train_brand3['day_of_week'] == 4]
df_train_brand3_week5 = df_train_brand3[df_train_brand3['day_of_week'] == 5]
df_train_brand3_week6 = df_train_brand3[df_train_brand3['day_of_week'] == 6]
df_train_brand3_week7 = df_train_brand3[df_train_brand3['day_of_week'] == 7]
plt.show()
plt.plot(df_train_brand3_week1.date, df_train_brand3_week1.cnt, 'b', label="point")
plt.savefig('pic/品牌3-星期1-趋势.png')
plt.show()
plt.plot(df_train_brand3_week2.date, df_train_brand3_week2.cnt, 'b', label="point")
plt.savefig('pic/品牌3-星期2-趋势.png')
plt.show()
plt.plot(df_train_brand3_week3.date, df_train_brand3_week3.cnt, 'b', label="point")
plt.savefig('pic/品牌3-星期3-趋势.png')
plt.show()
plt.plot(df_train_brand3_week4.date, df_train_brand3_week4.cnt, 'b', label="point")
plt.savefig('pic/品牌3-星期4-趋势.png')
plt.show()
plt.plot(df_train_brand3_week5.date, df_train_brand3_week5.cnt, 'b', label="point")
plt.savefig('pic/品牌3-星期5-趋势.png')
plt.show()
plt.plot(df_train_brand3_week6.date, df_train_brand3_week6.cnt, 'b', label="point")
plt.savefig('pic/品牌3-星期6-趋势.png')
plt.show()
plt.plot(df_train_brand3_week7.date, df_train_brand3_week7.cnt, 'b', label="point")
plt.savefig('pic/品牌3-星期7-趋势.png')

df_train_brand4_week1 = df_train_brand4[df_train_brand4['day_of_week'] == 1]
df_train_brand4_week2 = df_train_brand4[df_train_brand4['day_of_week'] == 2]
df_train_brand4_week3 = df_train_brand4[df_train_brand4['day_of_week'] == 3]
df_train_brand4_week4 = df_train_brand4[df_train_brand4['day_of_week'] == 4]
df_train_brand4_week5 = df_train_brand4[df_train_brand4['day_of_week'] == 5]
df_train_brand4_week6 = df_train_brand4[df_train_brand4['day_of_week'] == 6]
df_train_brand4_week7 = df_train_brand4[df_train_brand4['day_of_week'] == 7]
plt.show()
plt.plot(df_train_brand4_week1.date, df_train_brand4_week1.cnt, 'b', label="point")
plt.savefig('pic/品牌4-星期1-趋势.png')
plt.show()
plt.plot(df_train_brand4_week2.date, df_train_brand4_week2.cnt, 'b', label="point")
plt.savefig('pic/品牌4-星期2-趋势.png')
plt.show()
plt.plot(df_train_brand4_week3.date, df_train_brand4_week3.cnt, 'b', label="point")
plt.savefig('pic/品牌4-星期3-趋势.png')
plt.show()
plt.plot(df_train_brand4_week4.date, df_train_brand4_week4.cnt, 'b', label="point")
plt.savefig('pic/品牌4-星期4-趋势.png')
plt.show()
plt.plot(df_train_brand4_week5.date, df_train_brand4_week5.cnt, 'b', label="point")
plt.savefig('pic/品牌4-星期5-趋势.png')
plt.show()
plt.plot(df_train_brand4_week6.date, df_train_brand4_week6.cnt, 'b', label="point")
plt.savefig('pic/品牌4-星期6-趋势.png')
plt.show()
plt.plot(df_train_brand4_week7.date, df_train_brand4_week7.cnt, 'b', label="point")
plt.savefig('pic/品牌4-星期7-趋势.png')

df_train_brand5_week1 = df_train_brand5[df_train_brand5['day_of_week'] == 1]
df_train_brand5_week2 = df_train_brand5[df_train_brand5['day_of_week'] == 2]
df_train_brand5_week3 = df_train_brand5[df_train_brand5['day_of_week'] == 3]
df_train_brand5_week4 = df_train_brand5[df_train_brand5['day_of_week'] == 4]
df_train_brand5_week5 = df_train_brand5[df_train_brand5['day_of_week'] == 5]
df_train_brand5_week6 = df_train_brand5[df_train_brand5['day_of_week'] == 6]
df_train_brand5_week7 = df_train_brand5[df_train_brand5['day_of_week'] == 7]
plt.show()
plt.plot(df_train_brand5_week1.date, df_train_brand5_week1.cnt, 'b', label="point")
plt.savefig('pic/品牌5-星期1-趋势.png')
plt.show()
plt.plot(df_train_brand5_week2.date, df_train_brand5_week2.cnt, 'b', label="point")
plt.savefig('pic/品牌5-星期2-趋势.png')
plt.show()
plt.plot(df_train_brand5_week3.date, df_train_brand5_week3.cnt, 'b', label="point")
plt.savefig('pic/品牌5-星期3-趋势.png')
plt.show()
plt.plot(df_train_brand5_week4.date, df_train_brand5_week4.cnt, 'b', label="point")
plt.savefig('pic/品牌5-星期4-趋势.png')
plt.show()
plt.plot(df_train_brand5_week5.date, df_train_brand5_week5.cnt, 'b', label="point")
plt.savefig('pic/品牌5-星期5-趋势.png')
plt.show()
plt.plot(df_train_brand5_week6.date, df_train_brand5_week6.cnt, 'b', label="point")
plt.savefig('pic/品牌5-星期6-趋势.png')
plt.show()
plt.plot(df_train_brand5_week7.date, df_train_brand5_week7.cnt, 'b', label="point")
plt.savefig('pic/品牌5-星期7-趋势.png')

