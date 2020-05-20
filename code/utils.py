# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 10:28
# @Author  : Chen Yu

import os
import math
import numpy as np
import numpy.linalg as la
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
path = os.path.abspath('..')


def trainDataGen(raw_data, n_ts, n_day, n_week):
    data_x = []
    data_y = []

    date_ls = sorted(list(set(list(raw_data['date']))))
    # print(date_ls)
    # print(len(date_ls))
    date_ls_err = [20170504,20170508,20170509,20170616,20170627,20170628]
    date_ls_all = sorted(date_ls + date_ls_err)
    date_ls_train = []
    for date in date_ls_all:
        ls = []
        date_id = date_ls_all.index(date)
        if date_id < 7*n_week:
            continue
        for w in range(n_week):
            ls.append(date_ls_all[date_id - 7*(n_week-w)])
        for d in range(n_day):
            ls.append(date_ls_all[date_id - 1*(n_day-d)])
        ls.append(date)
        m = 0
        for i in ls:
            if i in date_ls:
                m +=1
        if m == 7:
            date_ls_train.append(ls)
    # print(date_ls_train)
    # print(len(date_ls_train))
    for i in range(len(date_ls_train)):
        day = date_ls_train[i][-1]
        start_ts_y = 64*date_ls.index(day)
        for t in range(64):
            yy = raw_data[raw_data[' timeslot']==start_ts_y+t]
            yyy = list(yy[' inFlow'])
            num_station = len(yyy)
            xxx = []
            for d in range(n_week+n_day):
                start_ts_x = 64*date_ls.index(date_ls_train[i][d])
                xx = raw_data[raw_data[' timeslot']==start_ts_x+t]
                xxx.append(list(xx[' inFlow']))
            for ts in range(n_ts):
                xx = raw_data[raw_data[' timeslot']==start_ts_y+t-n_ts+ts]
                xxx.append(list(xx[' inFlow']))

            for sta in range(num_station):
                xxxx = []
                yyyy = []
                for num_e in range(n_week+n_day+n_ts):
                    xxxx.append(xxx[num_e][sta])
                yyyy.append(yyy[sta])
                z = xxxx + yyyy
                if all(z):
                    data_x.append(xxxx)
                    data_y.append(yyyy)
                else:
                    continue
        if (i+1) % 10 == 0:
            print('dealed:', i+1, 'all:', len(date_ls_train))
        # if i+1 == 10:
        #     break
    return data_x, data_y


def MaxMinNorm(data, Max=None, Min=None):
    data = np.array(data)
    if Max == None:
        Max = np.max(data)
    else:
        Max = np.array(Max)
    if Min == None:
        Min = np.min(data)
    else:
        Min = np.array(Min)
    data = (data - Min)/(Max - Min)
    return data, Max, Min


def MaxMinNorm_re(data, Max_old, Min_old):
    data = data*(Max_old - Min_old) + Min_old
    return data


def evaluation(y_true, y_pred):
    y_true = y_true.numpy().reshape(-1, 1)
    y_pred = y_pred.numpy().reshape(-1, 1)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred))/(y_true + 1e-3)) * 100
    F_norm = la.norm(y_true - y_pred, 'fro') / (la.norm(y_true, 'fro') + 1e-3)
    # r2 = 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
    # var = 1 - (np.var(y_true - y_pred)) / np.var(y_true)
    return rmse, mae, mape, 1-F_norm


if __name__ == '__main__':
    # a = torch.ones(5)
    # b = torch.zeros(5)
    # print(evaluation(a, b))

    print('reading data ...')
    infile = path + '/data/raw_data/metroData_ODflow_15.csv'
    raw_flow = pd.read_csv(infile)
    N_TS = 3
    N_DAY = 3
    N_WEEK = 3

    trainData_x, trainData_y = trainDataGen(raw_flow, N_TS, N_DAY, N_WEEK)

    data_df = pd.DataFrame(trainData_x)
    data_df[9] = np.array(trainData_y).reshape(-1)

    data_df.to_csv(path + '/data/true_data/all_station_train_data.csv', header=None, index=None)

    read_df = pd.read_csv(path + '/data/true_data/all_station_train_data.csv', header=None)
    X = read_df.iloc[:, :-1].to_numpy()
    y = read_df.iloc[:, -1].to_numpy()