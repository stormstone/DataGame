原作链接：https://github.com/thuwyh/tianchi-shangpai-solution?spm=5176.9876270.0.0.182a2ef1PincPH

# 印象盐城·数创未来大数据竞赛 - 盐城汽车上牌量预测题解

初赛排名34，复赛排名25

比赛链接：https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.315ed780bOuAeJ&raceId=231641

首先恭喜排在前面的队伍，第一名两万多的mse真的很amazing，希望有一天能看到他们的方案，共同进步。

## 基本思路
- 没有把这道题当做时间序列来做，而是试图提取每一个日期对应的特征
- 首先需要还原出真实日期，因为很多特征是依据真实日期的
- 模型方面使用lightgbm、catboost和extratree，结合ensamble和bagging
- cv调参，在复赛b榜的时候用了15年10月11到16年10月11一整年作为验证集

## 使用的特征
1. 'brand', 品牌
1. 'day_of_week', 一周第几天
1. 'gap', 距离上一个date的天数
1. 'month', 月份
1. 'year', 年
1. 'week', 一年中的第几周
1. 'day', 一月中的第几天
1. 'xun', 一月中的第几周
1. 'lunar_month', 农历月
1. 'lunar_day', 农历日
1. 'lunar_year', 农历年
1. 'lunar_xun', 农历月中的第几周
1. 'last_year', 去年
1. 'last_lunar_year', 上个农历年
1. 's_mean', 去年同月的同品牌上牌平均
1. 's_skew', 去年同月的同品牌上牌偏度
1. 's_count', 去年同月的同品牌上牌数据个数
1. 'sx_mean', 去年同月的同周的同品牌上牌平均
1. 'sx_median', 去年同月的同周的同品牌上牌中位数
1. 'sx_max', 去年同月的同周的同品牌上牌最大值
1. 'sx_min', 去年同月的同周的同品牌上牌最小值
1. 'sx_count', 去年同月的同周的同品牌上牌数据个数
1. 'sx_std', 去年同月的同周的同品牌上牌标准差
1. 'l_mean', （农历）去年同月的同品牌上牌平均
1. 'l_median', 下面类似，我就不一一写了
1. 'l_count',
1. 'l_skew',
1. 'lx_mean',
1. 'lx_median',
1. 'lx_max',
1. 'lx_min',
1. 'lx_count',
1. 'lx_std',
1. 'hld', 是否假期
1. 'yesterday_hld', 昨天是否假期
1. 'tomorrow_hld', 明天是否假期
1. 'month_type', 月份分类
1. 'brand_type', 品牌分类
1. 'ith_workday', 是连续工作日中的第几个工作日，例如10月8日到10月18日都不是假日，那10月9日的特征值为2
1. 'ith_workday_r', 是连续工作日中的倒数第几个工作日，例如10月8日到10月18日都不是假日，那10月17日的特征值为2
1. 'spring_distance', 距离当年大年初一有几天
1. 'ld_distance', 距离当年劳动节有几天
1. 'nd_distance', 距离当年国庆节有几天
1. 'ny_distance', 距离当年元旦有几天
1. 'ny_distance2', 距离明年元旦有几天
1. 's_date_cnt'，去年在日期特性上与当前日期最接近的那天同品牌的上牌量

## 一些没有深入的思路
- kaggle前不久结束了一个餐厅访问量预测的比赛，跟这题非常类似，但是他最后预测的时间区间只有39天，第一名的队伍公开了他们的方案，非常震撼。我的理解他们的基本思路是通过窗口平移倍增数据，但天池这个比赛预测的区间远比他们长，似乎不太合适。https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/49129
- 复赛b榜预测的目标是16年10月12日到17年11月28日，整整一年多。上面方法会出现一些问题，就是17年的10月和11月没有去年的数据，我这次偷懒直接用15年的替换了，但其实更好的做法可能是先利用短期预测获得一个16年10月和11月的数据，也许会对成绩有帮助。
- 最后一天的时候我尝试使用人工系数来修正一下我的预测结果，没想到比上一次还差。乘系数比较主观，但到最后可能就是这些主观的因素决定最后的名次，who knows。 

## github工程
https://github.com/thuwyh/tianchi-shangpai-solution

- final_script: 主脚本
- holiday_v2.csv: 节假日数据
- data_script: 用于生成data_feature.csv的脚本

## 感谢
农历转换部分用到了一个开源项目https://github.com/isee15/Lunar-Solar-Calendar-Converter



