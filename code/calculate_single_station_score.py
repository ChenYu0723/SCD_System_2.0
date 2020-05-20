# -*- coding: utf-8 -*-
# @Time    : 2019/12/15 20:58
# @Author  : Chen Yu

import os
import numpy as np
import pandas as pd

os.chdir('..')

infile = 'result/single_station_result/single_station_score.csv'

df = pd.read_csv(infile, index_col=0)
print(df)

df2 = df[df.iloc[:, -1] != -1]
df2.mean()
'''
rmse           29.818689
mae            21.462222
mape           12.461562
acc             0.912005
'''