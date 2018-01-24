# 天池-新人实战赛之[离线赛]-商品个性化推荐

天池竞赛地址：[https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100068.5678.1.375de58bi3QXmm&raceId=231522](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100068.5678.1.375de58bi3QXmm&raceId=231522)

代码来源：https://github.com/chenkkkk/TianChi_YiDongTuiJian_forecast

数据来源：[https://tianchi.aliyun.com/getStart/information.htm?spm=5176.100067.5678.2.1c6d07a1VVnSTt&raceId=231522](https://tianchi.aliyun.com/getStart/information.htm?spm=5176.100067.5678.2.1c6d07a1VVnSTt&raceId=231522)

运行说明：

1.run ./Preprocess/Drop_Day_and_sub_item.py
2.run ./feature/extract_feture.py

线上成绩：F1评分：10.35332786%，准确率：0.09000000。

## 赛题描述：


**竞赛题目**

在真实的业务场景下，我们往往需要对所有商品的一个子集构建个性化推荐模型。在完成这件任务的过程中，我们不仅需要利用用户在这个商品子集上的行为数据，往往还需要利用更丰富的用户行为数据。定义如下的符号：
U——用户集合
I——商品全集
P——商品子集，P ⊆ I**D——用户对商品全集的行为数据集合
那么我们的目标是利用D来构造U中用户对*P*中商品的推荐模型。


**数据说明**
本场比赛提供20000用户的完整行为数据以及百万级的商品信息。竞赛数据包含两个部分。

第一部分是用户在商品全集上的移动端行为数据（D）,表名为tianchi_fresh_comp_train_user_2w

第二个部分是商品子集（P）,表名为tianchi_fresh_comp_train_item_2w


训练数据包含了抽样出来的一定量用户在一个月时间（11.18~12.18）之内的移动端行为数据（D），评分数据是这些用户在这个一个月之后的一天（12.19）对商品子集（P）的购买数据。参赛者要使用训练数据建立推荐模型，并输出用户在接下来一天对商品子集购买行为的预测结果。 


**评分数据格式**
具体计算公式如下：参赛者完成用户对商品子集的购买预测之后，需要将结果放入指定格式的数据表（非分区表）中，要求结果表名为：tianchi_mobile_recommendation_predict.csv，且以utf-8格式编码；包含user_id和item_id两列（均为string类型）,要求去除重复。


**评估指标**

比赛采用经典的精确度(precision)、召回率(recall)和F1值作为评估指标。具体计算公式如下：

![img](https://gtms01.alicdn.com/tps/i1/TB1WNN4HXXXXXbZaXXXwu0bFXXX.png)

其中PredictionSet为算法预测的购买数据集合，ReferenceSet为真实的答案购买数据集合。我们以F1值作为最终的唯一评测标准。

 