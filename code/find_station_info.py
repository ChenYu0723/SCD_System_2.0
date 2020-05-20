# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 15:49
# @Author  : Chen Yu

import os
import pickle
import pandas as pd
os.chdir('..')

infile2 = 'data/raw_data/transferStations.pkl'
transferStation_info = pickle.load(open(infile2, 'rb'))
infile3 = 'data/raw_data/metroStations.csv'
station_id_df = pd.read_csv(infile3)

# print(transferStation_info[0][2040])

# ==== 输出整合后站点信息
transferStation_info[0].keys()
df = pd.DataFrame(transferStation_info[0]).T
df.columns = ['包含站点', '站点名称']
df.to_csv('result/transferStation_info.csv', encoding='utf-8')