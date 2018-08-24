"""
date:2018-7-31
function:在网易财经上抓取股价信息(仅选取2016年举办的业绩说明会信息）
author：susuxuer
"""
import urllib.request
import re
import csv
import time

def get_wenben(path):
    csvfile = open(path, 'r', encoding='UTF-8')
    reader = csv.reader(csvfile)
    return reader

def get_all_code(): #获取全部2016举办业绩说明会的企业代码,时间
    path2 = 'D:/GFZQ/GFZQ/project/7_30_test/data/train/last_train.csv'
    reader2 = get_wenben(path2)
    return reader2

def get_com_information(): #获取业绩说明会时间，简称，全称，股票代码
    reader2 = get_all_code()
    companies =[]
    for item in reader2:
        companies.append(item)
    return companies

# 获取股票代码列表
def urlTolist(companies):
    allCodeList =[]
    allCompanyList =[]
    for item in companies:
        allCodeList.append(item[3])
        allCompanyList.append(item[1])
        #print (item)
    return allCodeList

def get_price(allCodelist):
    for code in allCodelist:
        print('正在获取%s股票数据...' % code)
        if code:
            if code[0] == '6':
                url = 'http://quotes.money.163.com/service/chddata.html?code=0' + code + \
                      '&end=20180413&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
            else:
                url = 'http://quotes.money.163.com/service/chddata.html?code=1' + code + \
                      '&end=20180413&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP'
            urllib.request.urlretrieve(url, 'D:/GFZQ/GFZQ/project/7_30_test/data/train/code/' + code + '.csv')  # 可以加一个参数dowmback显示下载进度

if __name__=='__main__':
    start = time.clock()
    companies  = get_com_information()
    allCodelist = urlTolist(companies)
    get_price(allCodelist)
    print("TimeUse:%s" %(time.clock()-start))
