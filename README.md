# fund_predict
尝试使用神经网络预测基金数据。该方法已经失败。无法有效预测。
# 通过神经网络寻找最大收益策略
基本方案：输入为基金过去历史净值数据，输出为评分权重。<br>
损失函数：以买入后81天内净值均值除以买入时净值作为涨跌幅，经过评分权重加权和KL散度约束，最大化收益率。设<img src="http://latex.codecogs.com/gif.latex?T_{i,j}" />
为第i个基金买入后第j日的历史净值，<img src="http://latex.codecogs.com/gif.latex?W_{i}" />为模型赋予该基金的权重则
<img src="http://latex.codecogs.com/gif.latex? Loss=\sum_{i}{w_i\sum_{j=0}^{80}{T_{i,j}/T_{i,0}}}+KL(W,N(0,1))" /><br>
模型整体架构：4层卷积。（LSTM与transformer尝试中...）
# 使用方法
首先通过爬虫爬取当日基金净值数据，然后筛选出时间较长的基金，然后训练
```sh
cd ./data/
python spider.py
python split.py
cd ..
python train.py
python test.py
```
