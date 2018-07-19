# -*- coding: utf-8 -*-
"""
author = 'suxuer'
date =
"""
import jieba
import jieba.posseg as pseg
import sys
import importlib

print("加载用户词典...")
importlib.reload(sys)
# sys.setdefaultencoding('utf8')     #py没有这个

# 下载需要分词的文件
jieba.load_userdict('G:/project/sentimation_analysis/dict/emotion_dict/pos_all_dict.txt')
jieba.load_userdict('G:/project/sentimation_analysis/dict/emotion_dict/neg_all_dict.txt')


# 分词，返回List
def segmentation(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = []
    for word in seg_list:
        seg_result.append(word)
    return seg_result


# 分词，词性标注，词和词性构成一个元组
def postagger(sentence):
    pos_data = pseg.cut(sentence)
    pos_list = []
    for w in pos_data:
        pos_list.append((w.word, w.flag))
    print(pos_list[:])
    return pos_list


# 句子切分
def cut_sentence(words):
    # words = words.decode('utf8')
    start = 0
    i = 0
    token = 'meaningless'
    sents = []
    # 根据标点符号将句子切分
    punt_list = ',.!?;~，。！？? ？；～… ：: ;'  # .decode('utf8')
    # print "punc_list", punt_list
    for word in words:
        # print "word", word
        if word not in punt_list:  # 如果不是标点符号
            # print "word1", word
            i += 1
            token = list(words[start:i + 2]).pop()
        # print "token:", token
        elif word in punt_list and token in punt_list:  # 处理省略号
            # print "word2", word
            i += 1
            token = list(words[start:i + 2]).pop()
        # print "token:", token
        else:
            # print "word3", word
            sents.append(words[start:i + 1])  # 断句
            start = i + 1
            i += 1
    if start < len(words):  # 处理最后的部分
        sents.append(words[start:])
    return sents


def read_lines(filename):
    fp = open(filename, 'r', encoding='UTF-8')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        line = line  # .decode("utf-8")
        lines.append(line)
    fp.close()
    return lines


# 去除停用词
def del_stopwords(seg_sent):
    stopwords = read_lines('G:/project/sentimation_analysis/data/stopwords.csv')  # 读取停用词表
    new_sent = []  # 去除停用词后的句子
    for word in seg_sent:
        if word in stopwords:
            continue
        else:
            new_sent.append(word)
    return new_sent


# 获取六种权值的词，根据要求返回list，这个函数是为了配合Django的views下的函数使用
def read_quanzhi(request):
    result_dict = []
    if request == "one":
        result_dict = read_lines("G://project/python_pachong/src/degree_dict/most.txt")
    elif request == "two":
        result_dict = read_lines("G://project/python_pachong/src/degree_dict/very.txt")
    elif request == "three":
        result_dict = read_lines("G://project/python_pachong/src/degree_dict/more.txt")
    elif request == "four":
        result_dict = read_lines("G://project/python_pachong/src/degree_dict/ish.txt")
    elif request == "five":
        result_dict = read_lines("G://project/python_pachong/src/degree_dict/insufficiently.txt")
    elif request == "six":
        result_dict = read_lines("G://project/python_pachong/src/degree_dict/inverse.txt")
    else:
        pass
    return result_dict


# seg_result = segmentation(test_sentence3)  # 分词，输入一个句子，返回一个list
# for w in seg_result:
# 	print (w)
# print ('\n')
# """
# """
# new_seg_result = del_stopwords(seg_result)  # 去除停用词
# for w in new_seg_result:
# 	print (w)
#
# postagger(test_sentence1)  # 分词，词性标注，词和词性构成一个元组
# cut_sentence(test_sentence2)    # 句子切分
# lines = read_lines("G://project/python_pachong/src/test_data.txt")
# print (lines[:])
