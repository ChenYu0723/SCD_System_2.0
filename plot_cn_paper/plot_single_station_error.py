# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 12:51
# @Author  : Chen Yu

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

os.chdir('..')

infile = 'result/single_station_result/single_station_score.csv'

df = pd.read_csv(infile, index_col=0)
# print(df)

df2 = df[df.iloc[:, -2] != -1]
df2.mean()
num_station = len(df2)
'''
rmse           29.818689
mae            21.462222
mape           12.461562
acc             0.912005
'''

x = range(1, num_station + 1)
y = df2['acc']
# ==== plot
plt.scatter(x, y, c='r', label='各站点')  # ls='--', marker='^',
# plt.plot(range(1, EPOCH + 1), rmse_test_all_ls[:EPOCH], c='b', label='测试集')  # ls='--', marker='^',
plt.axhline(y=y.mean(), color='b', ls='--', lw=2, label='均值')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.ylim((0, 100))
plt.xlabel('地铁站点', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
# plt.title('Rmse of all station')
plt.legend(loc='best', fontsize=14)
plt.tight_layout()
plt.savefig(r'result/single_station_result/acc_single_station.png', dpi=300)
plt.show()
