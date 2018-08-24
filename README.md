## emotion_analysis
  emotion analysis for earning communication Conference
  
                            |--get_conference   #在全景网上爬取业绩说明会文本信息
       |---    1.get_source|
       |                    |--get_priceCode    #在网易财经获取股票代码信息
       |
       |                    |--cal_emotion.py   #计算情感倾向，提取语调值
       |---    2.cal_emotion|
    scr|                    |--processing.py    #预处理
       |
       |                        |--tushare     #根据股票代码抓取股价信息
       |---    3.cal_flucutation|
       |                        |--cal_flucutation.py   #计算股价波动幅度
       |
       |                        |--                   # 相关性计算（找出情感与股价波动相关性最强的时间段）
       |---    4.cal_relation   |--                   # BP神经网络模型预测
       |                        |--                   #计算股价波动幅度与情感的关系
       |
       |                        |--model            #训练模型
       |---    5.cal_similarity |
       |                        |--similarity.py   #计算投资者提问和管理者回答的相似程度
