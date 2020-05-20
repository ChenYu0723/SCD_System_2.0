# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
import datetime
import random
from random import choice
import pickle

starttime = datetime.datetime.now()

# pd.set_option('display.height',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)

print('system start')
print('reading data ...')
# ====
infile = r'G:\Program\Pycharm Projects\File of Python3\SCD_System_2.0\data\raw_data\SH_metro_reduced.csv'
raw_data = pd.read_csv(infile, nrows=0)
# print(raw_data)
# raw_data.to_csv(r'C:\Program Files\Pycharm Projects\SCD_System_2.0\data\raw_data\SH_metro_reduced_2.csv',index=False)
infile2 = r'G:\Program\Pycharm Projects\File of Python3\SCD_System_2.0\data\raw_data\transferStations.pkl'
tfStations = pickle.load(open(infile2, 'rb'))
# print(tfStations)
print(tfStations[0])
# print(tfStations[1])
infile3 = r'G:\Program\Pycharm Projects\File of Python3\SCD_System_2.0\data\raw_data\metroStations.csv'
stationID_df = pd.read_csv(infile3)
# print(stationID_df)

'''
def time2ts(data):
    t = data['transTime']
    ts_ls = []
    for i in t:
        h = i / 10000
        m = (i - h * 10000) / 100
        ts = int(math.ceil((h * 60 + m) / 30.0))
        ts_ls.append(ts)
    data['timeSlot'] = ts_ls
    return data

def realStation(data):
    data['real_inStation'] = data['inStation']
    data['real_outStation'] = data['outStation']
    for i in range(len(data)):
        if data.iloc[i,5] in tfStations[1]:
            data.iloc[i,8] = tfStations[1][data.iloc[i,5]]
        else:
            pass
        if data.iloc[i,6] in tfStations[1]:
            data.iloc[i,9] = tfStations[1][data.iloc[i,6]]
        else:
            pass
    return data


print('initialize flow dataframe...')
stationID_ls = stationID_df['stationID']
date_ls = range(20150706,20150716)
ts_ls = range(11,49)
col = ['stationID', 'transDate', 'timeSlot', 'inFlow', 'outFlow']

stationFlow_df = pd.DataFrame(columns=col)
for id in stationID_ls:
    df_2 = pd.DataFrame(columns=col)
    for date in date_ls:
        df_1 = pd.DataFrame(columns=col)
        df_1['timeSlot'] = ts_ls
        df_1['transDate'] = date
        df_2 = df_2.append(df_1,ignore_index=True)
    df_2['stationID'] = id
    stationFlow_df = stationFlow_df.append(df_2, ignore_index=True)
stationFlow_df['inFlow'] = 0
stationFlow_df['outFlow'] = 0


print('deal...')
with open(infile, 'r') as f:
    rowsCount = 0
    userCount = 0
    errorCount = 0
    while 1:
        line = f.readline().rstrip().split(',')
        rowsCount += 1
        if rowsCount==1:
            col = line
            continue
        if rowsCount==2:
            pre_id = line[0]
        if pre_id != line[0]:
            userCount += 1
            time2ts(raw_data)
            realStation(raw_data)
            # print(raw_data)
            for i in range(len(raw_data)):
                a_in = raw_data.iloc[i,8]
                a_out = raw_data.iloc[i,9]
                b = raw_data.iloc[i,3]
                c = raw_data.iloc[i,7]
                if b not in date_ls or c not in ts_ls:
                    print(b)
                    print(c)
                    errorCount += 1
                    continue
                try:
                    row_in = list(stationID_ls).index(a_in) * len(date_ls) * len(ts_ls) + date_ls.index(b) * len(ts_ls) + ts_ls.index(c)
                    row_out = list(stationID_ls).index(a_out) * len(date_ls) * len(ts_ls) + date_ls.index(b) * len(ts_ls) + ts_ls.index(c)
                except ValueError:
                    print('!')
                stationFlow_df.iloc[row_in, 3] += 1
                stationFlow_df.iloc[row_out, 4] += 1
            # print('============================================')
            if userCount % 100 == 0:
                print('# deal user:',userCount)
            raw_data = pd.DataFrame(columns=col)
        if line == ['']:
            print('# rows:', rowsCount-2)
            print('# user:', userCount)
            print('# error:', errorCount)
            break
        # if userCount == 10000:
        #     break
        pre_id = line[0]
        line = [int(x) for x in line]
        raw_data.loc[rowsCount - 2] = line

# print(stationFlow_df)
stationFlow_df.to_csv(r'C:\Program Files\Pycharm Projects\SCD_System_2.0\data\flow_data\SH_metro_stationFlow.csv',index=False)
'''

print('system end')
endtime = datetime.datetime.now()
print('time:',(endtime - starttime).seconds,'s')
