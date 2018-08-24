import tushare as ts
import pandas as pd
import numpy as np
import datetime
import glob
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
    path1 = 'D:/GFZQ/GFZQ/data_/company_code/train_7_29.csv'
    path2 = 'D:/GFZQ/GFZQ/data_/2016all_train_7_30.csv'
    reader1 = get_wenben(path1)
    reader2 = get_wenben(path2)
    return reader1,reader2

def get_com_information(): #获取业绩说明会时间，简称，全称，股票代码
    return

# 获取公司股价信息
def get_price(code, date, days):
    # print()
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
    start_date = start.strftime('%Y-%m-%d')
    end_date = end.strftime('%Y-%m-%d')
    print(start_date)
    print(end_date)
    k = ts.get_hist_data(code, start=start_date, end=end_date)
    # print ( k.head(10) ) #查看前10行数据
    k = k.sort_index(axis=0, ascending=True)  # 对index 进行升序排列
    # 用移动平均法来进行预测
    lit = ['open', 'high', 'close', 'low']  # 这里我们只获取其中四列(最高、最低、开盘、闭盘)
    data = k[lit]
    # print (data)
    d_one = data.index  # 以下9行将object的index转换为datetime类型
    d_two = []
    d_three = []
    date2 = []
    for i in d_one:
        d_two.append(i)
    for i in range(len(d_two)):
        d_three.append(parse(d_two[i]))
    data2 = pd.DataFrame(data, index=d_three, dtype=np.float64)
    # 构建新的DataFrame赋予index为转换的d_three。当然你也可以使用date_range()来生成时间index
    length = len(data2['close'])
    ave_close = sum(data2['close']) / (length)
    last_day = data2['close'][-1]
    fluctuation = (last_day - ave_close) / ave_close  # 算术收益率
    print("average_price:%d,last_day_price:%d" % (ave_close, last_day))
    print('涨幅：%0.4f%%' % fluctuation)
    plt.plot(data2['close'])
    # 显然数据非平稳，所以我们需要做差分
    plt.title('股市每日收盘价')
    plt.show()

if __name__ == "__main__":
    code = '600548'
    start_date = '2016/10/30'
    days = [5, 10, 20, 60, 120, 250, 364]
    # days =31   #时间间隔
    for i in days:
        get_price(code, start_date, i)
    all_files = get_all_comp()
