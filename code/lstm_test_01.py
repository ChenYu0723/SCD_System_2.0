# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datetime import datetime
from lstm_train_02 import trainDataGen, MaxMinNorm, MaxMinNorm_re, mape, toV, LSTM

path = os.path.abspath('..')

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)


def main(STATION_ID):
    print('reading data ...')
    infile = path + '/data/true_data/metroData_ODflow_15.csv'
    raw_flow = pd.read_csv(infile)
    # print(raw_flow)
    station_flow = raw_flow[raw_flow[' station'] == STATION_ID]
    # print(station_flow)
    trainData_x, trainData_y = trainDataGen(station_flow, N_TS, N_DAY, N_WEEK)
    trainData_x, mao_x, mio_x = MaxMinNorm(trainData_x, 6864, 1)
    trainData_y, mao_y, mio_y = MaxMinNorm(trainData_y, 6864, 1)

    trainData_x = toV(trainData_x).view(len(trainData_x),1,N_TS+N_DAY+N_WEEK)
    trainData_y = toV(trainData_y).view(len(trainData_y),1,1)
    # print(trainData_x, len(trainData_x))
    # print(trainData_y, len(trainData_y))
    lll = len(trainData_x)
    test_x = trainData_x[int(0.75*lll):]
    test_y = trainData_y[int(0.75*lll):]
    # print(test_x, len(test_x))
    # print(test_y, len(test_y))

    # ==== restore net
    print('restoring net ...')
    lstm = LSTM(N_TS+N_DAY+N_WEEK, 10)
    lstm = torch.load('net_lstm_2.pkl')
    loss_func = nn.MSELoss()

    # ==== test
    print('testing data ...')
    lstm = lstm.eval()
    pred_outs = lstm(test_x)
    loss = loss_func(pred_outs, test_y)
    print('Test loss: {:.5f}'.format(loss.item()))

    # print(type(test_y), type(mao_y))
    # print(type(MaxMinNorm_re(test_y, mao_y, mio_y)))
    # mao_y = toV(mao_y)
    # mio_y = toV(mio_y)
    # test_y = MaxMinNorm_re(test_y, mao_y, mio_y).view(-1).data.numpy()
    # print(test_y)
    # print(all(test_y))
    # pred_outs = MaxMinNorm_re(pred_outs, mao_y, mio_y).view(-1).data.numpy()
    # mape_sta = mape(pred_outs, test_y)


    test_y = MaxMinNorm_re(test_y.view(-1).data.numpy(), 6864, 1)
    pred_outs = MaxMinNorm_re(pred_outs.view(-1).data.numpy(), 6864, 1)
    # print(test_y,type(test_y))
    mape_sta = mape(pred_outs, test_y)

    print('station id:', STATION_ID, 'mape:', mape_sta)

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(range(len(test_y)), test_y, ls='--', label='true data')
    # plt.plot(range(len(test_y)), pred_outs, label='prediction data')
    # plt.text(0, 0, 'Loss=%.5f' % loss.item(), fontdict={'size': 10, 'color': 'red'})
    # plt.title('Predict result of %s' % STATION_ID)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig(path + r'/result/station_flow_prediction_%s.png' % STATION_ID, dpi=150)

    # plt.show()
    return mape_sta


def main_2():
    infile3 = path + '/data/raw_data/metroStations.csv'
    station_id_df = pd.read_csv(infile3)
    # print(station_id_df)
    # mape_everySta_ls = []
    stationMape_df = pd.DataFrame(0, index=station_id_df['stationID'], columns=['mape'])
    # for i in range(len(station_id_df)):
    #     STATION_ID = station_id_df.iloc[i, 0]
    #     mape_sta = main(STATION_ID)
    #     mape_everySta_ls.append(mape_sta)
    print(stationMape_df)
    # stationMape_df.to_csv(path + r'/result/station_mape.csv')


def test():
    # infile = path + '/data/true_data/metroData_ODflow_15.csv'
    # raw_flow = pd.read_csv(infile)
    # # print(raw_flow)
    # station_flow = raw_flow[raw_flow[' station'] == STATION_ID]
    # # print(station_flow)
    # print('inFlow max:', max(station_flow[' inFlow']))
    # print('inFlow min:', min(station_flow[' inFlow']))

    # trainData_x, trainData_y = trainDataGen(station_flow, N_TS, N_DAY, N_WEEK)
    # trainData_x = MaxMinNorm(trainData_x)
    # trainData_y = MaxMinNorm(trainData_y)
    #
    # trainData_x = toV(trainData_x).view(len(trainData_x),1,N_TS+N_DAY+N_WEEK)
    # trainData_y = toV(trainData_y).view(len(trainData_y),1,1)
    # # print(trainData_x, len(trainData_x))
    # # print(trainData_y, len(trainData_y))
    # test_x = trainData_x[60*64:]
    # test_y = trainData_y[60*64:]
    # print(test_x, len(test_x))
    # print(test_y, len(test_y))

    infile2 = path + '/data/raw_data/transferStations.pkl'
    transferStation_info = pickle.load(open(infile2, 'rb'))
    # print(transferStation_info[0])
    # print(transferStation_info[0].keys())
    # print(transferStation_info[0].values())
    transferstation_df = pd.DataFrame(data=transferStation_info[0].values(),index=transferStation_info[0].keys(),columns=['includeStation','name'])
    print(transferstation_df)



if __name__ == '__main__':
    print('system start')
    starttime = datetime.now()

    # interval = 15
    # HOUR_INTERVAL = 16

    N_TS = 3
    N_DAY = 3
    N_WEEK = 3

    # STATION_ID = 1060  # 2011 2035 2010 2040 113 633
    # main(STATION_ID)
    main_2()
    # test()

    endtime = datetime.now()
    usetime = (endtime - starttime).seconds
    h = int(usetime / 3600)
    m = int((usetime - 3600 * h) / 60)
    s = usetime - 3600 * h - 60 * m
    print('time:', h, 'h', m, 'm', s, 's')
    print('system end')
