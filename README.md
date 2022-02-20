# fund_predict
尝试使用神经网络预测基金数据。
# 通过神经网络寻找最大收益策略
基本方案：输入为基金过去历史净值数据，输出为评分权重。<br>
损失函数：设<img src="http://latex.codecogs.com/gif.latex?T_{i,j}" />
为买入后第i个基金第j日的历史净值，<img src="http://latex.codecogs.com/gif.latex?W_{i}" />为模型赋予该基金的权重则
<img src="http://latex.codecogs.com/gif.latex? Loss=\frac{\sum_{i}{w_i\sum_{j=0}^{80}{T_{i,j}/T_{i,0}}}}{\sum_{i}w_i}" /><br>
模型整体架构：4层卷积。（LSTM与transformer尝试中...）
# 使用方法
首先通过爬虫爬取当日基金净值数据，并且划分为两个部分。然后在两个数据集上分别训练模型并且进行测试
```sh
cd ./data/
python spider.py
python split.py
cd ..
python train.py
python test.py
```
