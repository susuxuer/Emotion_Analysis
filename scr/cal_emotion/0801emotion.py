"""
计算情感值，并保存
2018/08/01--V1.0
by suxue
"""
import tushare as ts
import pandas as pd
import numpy as np
import datetime
import glob
import time
import csv
import re
import cal_emotion as ce
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_style("whitegrid", {"font.sans-serif": ['KaiTi', 'Arial']})

def get_all_comp():
    all_files = glob.glob(r'D:/GFZQ/GFZQ/xuesu2018/xuesu/*.csv')
    return all_files

def get_wenben(path):
    csvfile = open(path, 'r', encoding='UTF-8')
    reader = csv.reader(csvfile)
    return reader

def get_all_code(): #获取全部2016举办业绩说明会的企业代码,时间
    path2 = 'D:/GFZQ/GFZQ/project/7_30_test/data/train/2016all_train0731full.csv'
    reader2 = get_wenben(path2)
    return reader2

def get_com_information(): #获取业绩说明会时间，简称，全称，股票代码
    reader2 = get_all_code()
    companies =[]
    for item in reader2:
        companies.append(item)
    return companies

def getPath(companies):
    length = len(companies)
    path_all =[]
    for i in range(length):
        name = companies[i][2]
        path = 'D:/GFZQ/GFZQ/project/7_30_test/data/conferences/' + name + '.csv'
        path_all.append(path)
    return path_all

def getAllContext(allPath,length):
    allQ =[]
    allA =[]
    for i in range(length):
        path = allPath[i]
        f = open(path,encoding='UTF-8')
        df_data = pd.read_csv(f)
        question = df_data['Question']
        answer = df_data['Answer']
        allQ.append(question)
        allA.append(answer)
    return allQ,allA

def calEmotion1(singleQ,singleA):
    singleEmotion = []
    singleEmotion1 = []
    singleRes = []
    length =len(singleQ)
    #print (length )
    for i in range(length):
        ques = singleQ[i]
        ans =singleA[i]
        pos_result, neg_result, tonescore, res = ce.single_review_sentiment_score(ques)
        pos_result1, neg_result1, tonescore1, res1 = ce.single_review_sentiment_score(ans)
        singleEmotion.append([pos_result, neg_result])
        singleEmotion1.append([pos_result1, neg_result1])
        singleRes.append(res)
    # result1 = 0
    # result2 = 0
    # result1_ = 0
    # result2_ = 0
    # for item1, item2 in singleEmotion:
    #     result1 += item1
    #     result2 += item2
    # for item1_, item2_ in singleEmotion1:
    #     result1_ += item1_
    #     result2_ += item2_
    # pos1 = result1 - result2
    # #print ("pos1:%d" %pos1)
    # pos2 = result1 + result2
    # #print ("pos2%d"%pos2)
    # pos1_ = result1_ - result2_
    # pos2_ = result1_ + result2_
    # try:
    #     tone1 = pos1 / pos2                 #投资者语调
    #     #tone = round(result, 3)
    #     tone2 = pos1_ / pos2_               #管理者语调
    #     #tone = round(result, 3)
    #     #print("本次业绩说明会管理者的总体情感倾向:%f" % tone)
    # except Exception as e:
    #     tone1 = 0.0
    #     tone2 = 0.0
    #     #print("本次业绩说明会管理者的总体情感倾向:%f" % tone)


    #return tone1,tone2,result1,result2,singleRes
    return singleRes
def calEmotion2(question, answer,len1):
    allToneQ =[]
    allToneA =[]
    sentToneA =[]
    sentToneQ =[]
    allRes =[]
    for i in range(len1):
        print (i)
        #print (allPath[i])
        singleQ = question[i]      #一个文本中所有的问题

        #print (singleQ)
        singleA = answer[i]         #一个文本中所有的答案
        #计算情感
        #for j in range(len2) :
        #toneQ,toneA,singleEmotion,singleEmotion1,singleRes = calEmotion1(singleQ,singleA)  #每句话的语调
        singleRes = calEmotion1(singleQ,singleA)  #每句话的语调
        allRes.append(sum(singleRes))
        #print (len(singleRes))

        # allToneQ.append(singleEmotion)
        # allToneA.append(singleEmotion1)      #allToneA保存的是一整个文本的所有语调

    #return allToneQ,allToneA
    return allRes

#获取股票涨跌幅度
def get_price(code, date, days):
    year = date[:4]
    month = date[5:7]
    # print("month:%s" % month)
    day = date[8:10]
    # print ("day:%s"%day)
    date = re.sub('/', '-', date)
    now = datetime.datetime(int(year), int(month), int(day))
    delta = datetime.timedelta(days)
    start = now - delta
    end = now + delta  # 日期换算
    now = now.strftime('%Y-%m-%d')
    start_date = start.strftime('%Y-%m-%d')
    end_date = end.strftime('%Y-%m-%d')
    #print("from %s to %s"%(start_date,end_date))
    k1 = ts.get_hist_data(code, start= now, end=end_date)
    k2 = ts.get_hist_data(code, start=start_date, end=now)
    #print ( k.head(10) ) #查看前10行数据
    k1 = k1.sort_index(axis=0, ascending=True)  # 对index 进行升序排列
    k1 = k1.sort_index(axis=0, ascending=True)
    # 用移动平均法来进行预测
    lit = ['open', 'high', 'close', 'low']  # 这里我们只获取其中四列(最高、最低、开盘、闭盘)
    data1 = k1[lit]
    data2 = k2[lit]
    # print (data)
    d_one = data1.index  # 以下9行将object的index转换为datetime类型
    d_two = []
    d_three = []
    for i in d_one:
        d_two.append(i)
    for i in range(len(d_two)):
        d_three.append(parse(d_two[i]))
    data1_ = pd.DataFrame(data1, index=d_three, dtype=np.float64)
    d_one1 = data1.index  # 以下9行将object的index转换为datetime类型
    d_two1 = []
    d_three1 = []
    for i in d_one:
        d_two1.append(i)
    for i in range(len(d_two1)):
        d_three1.append(parse(d_two1[i]))
    data2_ = pd.DataFrame(data2, index=d_three1, dtype=np.float64)
    # 构建新的DataFrame赋予index为转换的d_three。当然你也可以使用date_range()来生成时间index
    length1 = len(data1_['close'])
    length2 = len(data2_['close'])
    try:
        ave_close1 = sum(data1_['close']) / (length1)
        ave_close2 = sum(data2_['close']) / (length2)
        last_day1 = data1_['close'][-1]
        last_day2 = data2_['close'][-1]
        fluctuation1 = (last_day1 - ave_close1) / ave_close1   #业绩说明会后days天的涨跌幅度
        fluctuation2 = (last_day2 - ave_close2) / ave_close2   #业绩说明会前days天的涨跌幅度
        change = (ave_close1 - ave_close2)/ave_close2
    except ZeroDivisionError as e:
        fluctuation1 = 0
        fluctuation2 =0
        change = 0

        #print("average_price:%d,last_day_price:%d" % (ave_close, last_day))
    #print('涨幅：%0.4f%%' % fluctuation)
    # plt.plot(data2['close'])
    # # 显然数据非平稳，所以我们需要做差分
    # plt.title('股市每日收盘价')
    # plt.show()
    return fluctuation1,fluctuation2,change

def getDifferentDay(all):
    day5 =[]
    day10 =[]
    day20=[]
    day60=[]
    day120 = []
    day250 =[]
    day364 =[]
    for item in all:
        day5.append(item[0])
        day10.append(item[1])
        day20.append(item[2])
        day60.append(item[3])
        day120.append(item[4])
        day250.append(item[5])
        day364.append(item[6])
    return day5,day10,day20,day60 ,day120,day250,day364

def savePriceChange(day5, day10, day20, day60, day120, day250, day364,date,short,full,code):
    predictions = ['date','code','short','full','day5','day10','day20','day60','day120','day250','day364']
       # pd.concat([answer_column, question_column], axis=1)
    df1 = pd.DataFrame({'date':date,'code':code,'short':short,'full':full,'day5':day5,'day10': day10,'day20':day20 ,'day60':day60,'day120':day120 ,'day250':day250,'day364':day364},columns=predictions)
    df1.to_csv('D:/GFZQ/GFZQ/project/7_30_test/data/priceChange/priceChange901_.csv', index=False)

def saveTone(companies,allToneQ,allToneA):
    short = []
    code =[]
    length = len(allToneQ)
    for i in range(length):
        short.append(companies[i][1])
        code.append(companies[i][3])
    predictions = ['code','short','allToneQ','allToneA']
    df1 = pd.DataFrame({'code':code,'short':short,'allToneQ':allToneQ,'allToneA':allToneA},columns=predictions)
    df1.to_csv('D:/GFZQ/GFZQ/project/7_30_test/data/Tone/tone0815Q.csv', index=False)

def saveS(code,short, allToneQ, allToneA ,allRes):
    predictions = ['code', 'short', 'allToneQ', 'allToneA','allRes']
    df1 = pd.DataFrame({'code': code, 'short': short, 'allToneQ': allToneQ, 'allToneA': allToneA,'allRes':allRes}, columns=predictions)
    df1.to_csv('D:/GFZQ/GFZQ/project/7_30_test/data/Tone/tone0815divall.csv', index=False)

def cal_time(timeUse):
    if timeUse <= 60:
        print ("timeUse:%ssec"%timeUse)
    if 60 < timeUse <=  60*60.0:
        print ("timeUse:%smins"%(timeUse/60.0))
    if 60*60.0 < timeUse <=  60*60*60.0:
        print ("timeUse:%shours"%(timeUse/(60*60.0)))

if __name__ == "__main__":
    start = time.clock()
    companies = get_com_information()
    allPath = getPath(companies)
    len1 = len(allPath)
    question, answer = getAllContext(allPath,len1)  #获取所有文本[[],[],[]]list
    allRes = calEmotion2(question, answer, len1)
    #print (allRes)
    df_tone = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/tone/posneg0815A.csv')
    allToneQ = df_tone['posA']
    allToneA  = df_tone['negA']
    code = df_tone['code']
    short = df_tone['short']
    #print (allToneA)
    #保存文件
    saveS(code,short, allToneQ, allToneA ,allRes)
    cal_time(time.clock()-start)

