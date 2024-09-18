'''
useage:
    train_data = pd.read_excel('LCQUAD.xlsx')
    processed_data = GetTarget(train_data)
'''

import numpy as np
import pandas as pd
import re
import math

#select SELECT DISTINCT
#ASK WHERE

Labletrans = {
    'S P ?ans ' : 'A',
    'S P ?x ' : 'a',
    '?ans P O ' : 'B',
    '?x P O ' : 'b',
    '?ans P ?x ' :'C',
    '?x P ?ans ' : 'c',
    'S P O ' : 'D',
    }

#train_data = pd.read_excel( "./Data/QALD9train.xlsx")
#train_data = pd.read_excel( "./Data/LCQUAD.xlsx",sheet_name='LCQUAD')

def GetTarget(train_data):
    col_name = train_data.columns.tolist()
    col_name.insert(2,'lable') 
    train_data = train_data.reindex(columns=col_name)
    
    query = train_data["query"].copy() #不改train_data
    parentheses = re.compile(r'[{](.*)[}]', re.S)    
    for i in range(len(query)):
        while 1 :
            if "".join(re.findall(parentheses, query[i])) == "":
                break
            else:
                query[i] = "".join(re.findall(parentheses, query[i])) #list to string
        query[i] = GetSPO(query[i])
        #query[i] = Delrdftype(query[i])
        query[i] = Convert_question_mark(query[i], train_data, i)
        get_target(train_data,query[i],i)
        train_data['lable'][i] = get_lable(train_data['lable'][i])
    return train_data

def GetSPO(subquery):
    temp = []
    count = 0
    subspo = subquery.split()
    for i in range(len(subspo)):
        if subspo[i] =="." or subspo[i] ==";":
            continue
        if "?" in subspo[i]:
            delperiod = ''
            for j in subspo[i]:
                if j != ".":
                    delperiod = delperiod+j
            temp.append(delperiod)
            count +=1
        else :
            if count%3 == 0:
                temp.append("S")
            if count%3 == 1:
                    temp.append("P")
            if count%3 == 2:
                temp.append("O")
            count +=1
    return temp

#沒用到
def Delrdftype(query):
    if "rdf:type" in query:
        #print(query.index("rdf:type"))
        typeloc = query.index("rdf:type")
        i = 0
        while i < 3:
            del query[typeloc-1]
            i +=1
    return query

def Getans(query):
    select_distinct = query.split()
    regular_ans = re.compile(r'[()](.*)[)]', re.S)
    if select_distinct[0] == 'SELECT' and select_distinct[1] == 'DISTINCT' and '?' in select_distinct[2]:
        if '(' in select_distinct[2]:
            select_distinct[2] = "".join(re.findall(regular_ans,select_distinct[2]))
        return select_distinct[2]
      
    #ans = train_data["query"][32]
    while 1:
        if "".join(re.findall(regular_ans,query)) == "":
            break
        query = "".join(re.findall(regular_ans,query))
    query = query.split()
    query = query[1]
    
    return query

def Convert_question_mark(query, train_data,index):
    temp = []
    ans_candidate = []
    for i in query:
        if '?' in i and i not in ans_candidate:
            ans_candidate.append(i)
    if len(ans_candidate) > 1:
        ans = Getans(train_data['query'][index])
        if "?" not in ans:
            ans = '?w'
    elif len(ans_candidate) == 1:
        ans = ans_candidate[0]
    for i in query:
        if '?' in i :
            if i == ans:
                temp.append('?ans')
            else:
                temp.append('?x')
        else:
            temp.append(i)
    return temp

def get_target(train_data,query,index):
    target = []
    k = 0
    for i in range(int(len(query)/3)):
        temp = ''
        for j in range(3):
            temp = temp +''.join(query[k])+' '
            k += 1
        #print(temp)
        target.append(Labletrans[temp])
    train_data['lable'][index] = target 

def get_lable(lable):
    # 定義映射規則
    mapping = {
        'A': ['A', 'E', 'H'],
        'a': ['a', 'e', 'h'],
        'B': ['B', 'F', 'I'],
        'b': ['b', 'f', 'i'],
        'C': ['C', 'G', 'J'],
        'c': ['c', 'g', 'j'],
        'D': ['D']
    }
    
    new_lable = []
    
    # 若 lable 為空，返回 ['d']
    if not lable:
        return ['d']
    
    for key, values in mapping.items():
        if key in lable:
            lable.remove(key)
            new_lable.append(values[0])
            values.pop(0)  # 移除已使用的第一個元素
    
    # 為剩餘的 label 指派對應的值
    for key, values in mapping.items():
        if key in lable:
            lable.remove(key)
            new_lable.append(values[0])
            values.pop(0)
    
    return new_lable

