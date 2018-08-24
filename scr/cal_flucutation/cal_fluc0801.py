import tushare as ts
import pandas as pd
import numpy as np
import datetime
import glob
import time
import csv
import re
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

def get_emotion():
    return

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
    now1 = now.strftime('%Y-%m-%d')
    start_date = start.strftime('%Y-%m-%d')
    end_date = end.strftime('%Y-%m-%d')
    #print("from %s to %s"%(start_date,end_date))
    k1 = ts.get_hist_data(code, start= now1, end=end_date)
    k2 = ts.get_hist_data(code, start=start_date, end=now1)
    # print ( k1.head(10) ) #查看前10行数据
    #print(k2.head(10))
    k1 = k1.sort_index(axis=0, ascending=True)  # 对index 进行升序排列
    k2 = k2.sort_index(axis=0, ascending=True)
    # 用移动平均法来进行预测
    lit = ['open', 'high','close', 'low']  # 这里我们只获取其中四列(最高、最低、开盘、闭盘)
    data1 = k1[lit]
    data2 = k2[lit]
    #print (data1)
    #print (data2)
    d_one = data1.index  # 以下9行将object的index转换为datetime类型
    d_two = []
    d_three = []
    for i in d_one:
        d_two.append(i)
    for i in range(len(d_two)):
        d_three.append(parse(d_two[i]))
    data1_ = pd.DataFrame(data1, index=d_three, dtype=np.float64)
    d_one1 = data2.index  # 以下9行将object的index转换为datetime类型
    d_two1 = []
    d_three1 = []
    for i in d_one1:
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
        #fluctuation = sum()
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
    df1.to_csv('D:/GFZQ/GFZQ/project/7_30_test/data/priceChange/priceChange0802900_.csv', index=False)

def cal_time(timeUse):
    if timeUse <= 60:
        print ("timeUse:%ssec"%timeUse)
    if 60 < timeUse <=  60*60.0:
        print ("timeUse:%smins"%(timeUse/60.0))
    if 60*60.0 < timeUse <=  60*60*60.0:
        print ("timeUse:%shours"%(timeUse/(60*60.0)))

if __name__ == "__main__":
    time1 = time.clock()
    date1 =[]
    short1 =[]
    full1 =[]
    code1 =[]
    companies = get_com_information()
    length = len(companies)
    allChange = []          #业绩说明会前后涨跌幅度
    allAfterChange =[]      #业绩说明会之后涨跌幅度
    for i in range(900,length):
        print (companies[i])
        date1.append(companies[i][0])
        short1.append(companies[i][1])
        full1.append(companies[i][2])
        code1.append(companies[i][3])

        code = companies[i][3]  #股票代码
        date = companies[i][0]  #业绩说明会举办时间
        priceChange = []
        changeAfterCon =[]
        days = [5, 10, 20, 60, 120, 250, 364]
        #days=[5]
        for j in days:
            try:
                fluctuation1, fluctuation2, change = get_price(code, date, j)
                #print (change)
                priceChange.append(change)
                changeAfterCon.append(fluctuation1)
                #print (companies[i][2],fluctuation)
            except AttributeError as e:
                fluctuation1 = 0
                change = 0
                priceChange.append(change)
                changeAfterCon.append(fluctuation1)
                #print ("%s has no code"%companies[i][1])
                #break
            #print (priceChange)
        allChange.append(priceChange)
        allAfterChange.append(changeAfterCon)
    day5, day10, day20, day60, day120, day250, day364 = getDifferentDay(allChange)
    #print (date1)
    #print (short1)
    savePriceChange(day5, day10, day20, day60, day120, day250, day364,date1,short1,full1,code1)
    cal_time(time.clock() - time1)



    ###test
    # code = '002029'
    # start_date = '2016/10/30'
    # days = [5, 10, 20, 60, 120, 250, 364]
    # days =31   #时间间隔
    # all =[]
    # for i in days:
    #     fluctuation1, fluctuation2, change =get_price(code, start_date, i)
    #     all.append(change)
    # for i in all:
    #     print (i)