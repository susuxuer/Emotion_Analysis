"""
合并数据
“将股价和情感，语调存放在一起”
"""
import pandas as pd

def get_wenben(path):
    csvfile = open(path, 'r', encoding='UTF-8')
    reader = csv.reader(csvfile)
    return reader

def savePriceChange(day5, day10, day20, day60, day120, day250, day364,date,short,full,code,toneA,toneQ):
    predictions = ['date','code','short','full','toneQ','toneA','day5','day10','day20','day60','day120','day250','day364']
       # pd.concat([answer_column, question_column], axis=1)
    df1 = pd.DataFrame({'date':date,'code':code,'short':short,'full':full,'toneQ':toneQ,'toneA':toneA,'day5':day5,'day10': day10,'day20':day20 ,'day60':day60,'day120':day120 ,'day250':day250,'day364':day364},columns=predictions)
    df1.to_csv('D:/GFZQ/GFZQ/project/7_30_test/data/priceChange/allToneChange0803.csv', index=False)

def saveSimilarity(day5, day10, day20, day60, day120, day250,day364,date,short,full,code,tfidf0,lsi0):
    predictions = ['date','code','short','full','tfidf0','lsi0','day5','day10','day20','day60','day120','day250','day364']
    df1 = pd.DataFrame({'date':date,'code':code,'short':short,'full':full,'tfidf0':tfidf0,'lsi0':lsi0,'day5':day5,'day10': day10,'day20':day20 ,'day60':day60,'day120':day120 ,'day250':day250,'day364':day364},columns=predictions)
    df1.to_csv('D:/GFZQ/GFZQ/project/7_30_test/data/similarity/allSimilarityChange0803.csv', index=False)

def saveAll(day5, day10, day20, day60, day120, day250,day364,date,short,full,code,tfidf0,lsi0,toneQ,toneA):
    predictions = ['date','code','short','toneQ','toneA','tfidf0','lsi0','day5','day10','day20','day60','day120','day250','day364']
    df1 = pd.DataFrame({'date':date,'code':code,'short':short,'toneQ':toneQ,'toneA':toneA,'tfidf0':tfidf0,'lsi0':lsi0,'day5':day5,'day10': day10,'day20':day20 ,'day60':day60,'day120':day120 ,'day250':day250,'day364':day364},columns=predictions)
    df1.to_csv('D:/GFZQ/GFZQ/project/7_30_test/data/similarity/allcat0803.csv', index=False)


def getData():
    df_price = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/priceChange/priceChangeAll0802.csv')
    df_similarity = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/similarity/similarityAll0801.csv')
    df_tone = pd.read_csv('D:/GFZQ/GFZQ/project/7_30_test/data/Tone/toneAll0801.csv')

    tfidf0 = df_similarity['tfidf0']
    lsi0 = df_similarity['lsi0']
    date =df_price['date']
    code = df_price['code']
    short = df_price['short']
    full = df_price['full']
    day5 = df_price['day5']
    day10 = df_price['day10']
    day20 = df_price['day20']
    day60 = df_price['day60']
    day120 = df_price['day120']
    day250 = df_price['day250']
    day364 = df_price['day364']
    toneQ = df_tone['allToneQ']
    toneA = df_tone['allToneA']
    #print (ToneA,ToneQ)
    savePriceChange(day5, day10, day20, day60, day120, day250, day364, date, short, full, code, toneA, toneQ)
    saveSimilarity(day5, day10, day20, day60, day120, day250, day364, date, short, full, code, tfidf0, lsi0)
    saveAll(day5, day10, day20, day60, day120, day250, day364, date, short, full, code, tfidf0, lsi0,toneQ,toneA)
if __name__=='__main__':
     getData()