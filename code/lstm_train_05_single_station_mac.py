# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 10:28
# @Author  : Chen Yu

import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from utils import *
from datetime import datetime
# 每个station单独训练model，epoch后测试剩余样本的指标，记录所有站点指标
# 这个文件是某个station训练完后保存其模型并计算其指标
plt.rcParams['font.sans-serif'] = ['SimHei']
path = os.path.abspath('..')

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)


# def mape(y_pred, y_true):
#     y_pred = np.array(y_pred)
#     y_true = np.array(y_true)
#     m = np.mean(np.abs((y_true - y_pred)/(y_true + 1e-3))) * 100
#     return m
#
#
# def mae(y_pred, y_true):
#     y_pred = np.array(y_pred)
#     y_true = np.array(y_true)
#     m = np.mean(np.abs(y_true - y_pred))
#     return m


def toV(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.out(x)
        x = x.view(s, b, -1)
        return x


def main(STATION_ID):
    print('reading data ...')
    infile = path + '/data/raw_data/metroData_ODflow_15.csv'
    raw_flow = pd.read_csv(infile)
    # print(raw_flow)

    # ==== only train one station model ====
    station_flow = raw_flow[raw_flow[' station'] == STATION_ID]
    # print(station_flow)
    # ====

    trainData_x, trainData_y = trainDataGen(station_flow, N_TS, N_DAY, N_WEEK)
    # trainData_xy = np.array([trainData_x, trainData_y])
    # trainData_xy_no, mao, mio = MaxMinNorm(trainData_xy)
    trainData_x, mao_x, mio_x = MaxMinNorm(trainData_x)
    trainData_y, mao_y, mio_y = MaxMinNorm(trainData_y)

    # trainData_x = trainData_xy_no[0]
    # trainData_y = trainData_xy_no[1]

    trainData_x = toV(trainData_x).view(len(trainData_x),1,N_TS+N_DAY+N_WEEK)
    trainData_y = toV(trainData_y).view(len(trainData_y),1,1)
    # print(trainData_x, len(trainData_x))
    # print(trainData_y, len(trainData_y))
    lll = len(trainData_x)
    train_x = trainData_x[:int(0.75*lll)]  # 60*64*322
    train_y = trainData_y[:int(0.75*lll)]
    # print(train_x, len(train_x))
    # print(train_y, len(train_y))
    test_x = trainData_x[int(0.75*lll):]
    test_y = trainData_y[int(0.75*lll):]

    num_train = len(train_x)

    trainData_set = Data.TensorDataset(train_x, train_y)
    trainData_loader = Data.DataLoader(
        dataset=trainData_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # testData_set = Data.TensorDataset(test_x, test_y)
    # testData_loader = Data.DataLoader(
    #     dataset=testData_set,
    #     batch_size=BATCH_SIZE_TEST,
    #     shuffle=True,
    #     num_workers=2
    # )


    lstm = LSTM(N_TS+N_DAY+N_WEEK, 10)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

    # ==== train
    print('training data ...')
    mape_all_ls = []
    # mape_test_all_ls = []
    for epoch in range(EPOCH):
        print('epoch:', epoch+1)
        mape_epoch_ls = []
        for step, (batch_x, batch_y) in enumerate(trainData_loader):
            outs = lstm(batch_x)
            loss = loss_func(outs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 100 == 0:
                print(' step:', step+1, 'all:', int(num_train/BATCH_SIZE))

            y_pred = MaxMinNorm_re(outs, mao_y, mio_y).view(-1).data
            y_true = MaxMinNorm_re(batch_y, mao_y, mio_y).view(-1).data
        #     mape_step = mape(y_pred, y_true)
        #     mape_epoch_ls.append(mape_step)
        # mape_all_ls.append(np.mean(mape_epoch_ls))
        if (epoch+1) % 10 == 0:
            print('epoch: {}, loss: {:.5f}'.format(epoch+1, loss.item()))

        # for step_test, (batch_x_test, batch_y_test) in enumerate(testData_loader):
        #     outs_test = lstm(batch_x_test)
        #     y_pred_test = MaxMinNorm_re(outs_test, mao_y, mio_y).view(-1).data
        #     y_true_test = MaxMinNorm_re(batch_y_test, mao_y, mio_y).view(-1).data
        #     mape_step_test = mape(y_pred_test, y_true_test)
        #     mape_test_all_ls.append(mape_step_test)
        #     break

    # ==== test
    pred_outs = lstm(test_x)
    pred_outs = MaxMinNorm_re(pred_outs, mao_y, mio_y).view(-1).data
    test_y = MaxMinNorm_re(test_y, mao_y, mio_y).view(-1).data

    rmse_sta, mae_sta, mape_sta, acc_sta = evaluation(test_y, pred_outs)

    print('station id:', STATION_ID, 'rmse:', rmse_sta)
    print('station id:', STATION_ID, 'mae:', mae_sta)
    print('station id:', STATION_ID, 'mape:', mape_sta)
    print('station id:', STATION_ID, 'acc:', acc_sta)


    # plt.plot(range(1, EPOCH+1), mape_all_ls, c='r', label='Training Mape')  # ls='--', marker='^',
    # plt.plot(range(1, EPOCH+1), mape_test_all_ls, c='b', label='Testing Mape')  # ls='--', marker='^',
    # # plt.xticks(range(1, EPOCH+1))
    # # plt.yticks(range(0, 110, 10))
    # # plt.ylim((0, 100))
    # plt.xlabel('Epoch')
    # plt.ylabel('Mape (%)')
    # plt.title('Mape of station-%s' % STATION_ID)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig(path + r'/result/mape_%s.png' % STATION_ID, dpi=150)
    # plt.show()

    # ==== save model
    print('saving net ...')
    print('mao_x:', mao_x)  # mao_x: 6864
    print('mio_x:', mio_x)  # mio_x: 1
    print('mao_y:', mao_y)  # mao_y: 6864
    print('mio_y:', mio_y)  # mio_y: 1
    torch.save(lstm, path + r'/model/single_station_model/net_lstm_%s.pkl' % STATION_ID)

    return rmse_sta, mae_sta, mape_sta, acc_sta


def main_2():
    infile3 = path + '/data/raw_data/metroStations.csv'
    station_id_df = pd.read_csv(infile3)
    station_id_df['rmse'] = -1
    station_id_df['mae'] = -1
    station_id_df['mape'] = -1
    station_id_df['acc'] = -1
    # print(station_id_df)
    station_num = len(station_id_df)
    for i in range(station_num):
        sta_id = station_id_df.iloc[i, 0]
        try:
            rmse_sta, mae_sta, mape_sta, acc_sta = main(sta_id)
            station_id_df.iloc[i, 3] = rmse_sta
            station_id_df.iloc[i, 4] = mae_sta
            station_id_df.iloc[i, 5] = mape_sta
            station_id_df.iloc[i, 6] = acc_sta
        except:
            pass

        print('dealed:', i+1, 'all:', station_num)
        station_id_df.to_csv(path + r'/result/single_station_score.csv')
    # print(station_id_df)


def test():
    infile = path + '/data/raw_data/metroData_ODflow_15.csv'
    raw_flow = pd.read_csv(infile)
    # print(raw_flow)
    print('inFlow max:', max(raw_flow[' inFlow']))  # inFlow max: 6864
    print('inFlow min:', min(raw_flow[' inFlow']))  # inFlow min: 0

    # trainData_x, trainData_y = trainDataGen(raw_flow, N_TS, N_DAY, N_WEEK)
    # trainData_x = MaxMinNorm(trainData_x)
    # trainData_y = MaxMinNorm(trainData_y)
    #
    # trainData_x = toV(trainData_x).view(len(trainData_x),1,N_TS+N_DAY+N_WEEK)
    # trainData_y = toV(trainData_y).view(len(trainData_y),1,1)
    # print(trainData_x, len(trainData_x))
    # print(trainData_y, len(trainData_y))

    # trainData_set = Data.TensorDataset(trainData_x, trainData_y)
    # trainData_loader = Data.DataLoader(
    #     dataset=trainData_set,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=2
    # )
    #
    # for epoch in range(1):
    #     for step, (batch_x, batch_y) in enumerate(trainData_loader):
    #         print('Epoch: ', epoch, '| Step: ', step)
    #         print('batch x: ')
    #         print(batch_x)
    #         print('batch y: ')
    #         print(batch_y)


if __name__ == '__main__':
    print('system start')
    starttime = datetime.now()

    # interval = 15
    # HOUR_INTERVAL = 16
    # STATION_ID = 2011  # 2035 2011

    N_TS = 3
    N_DAY = 3
    N_WEEK = 3
    EPOCH = 50
    BATCH_SIZE = 100
    # BATCH_SIZE_TEST = 5000
    LR = .01

    main(2040)
    # main_2()
    # test()

    endtime = datetime.now()
    usetime = (endtime - starttime).seconds
    h = int(usetime / 3600)
    m = int((usetime - 3600 * h) / 60)
    s = usetime - 3600 * h - 60 * m
    print('time:', h, 'h', m, 'm', s, 's')
    print('system end')

'''
113：莲花路
247:陆家嘴
2010：世纪大道
2011：莘庄
2035：人民广场
2040：东方体育中心
'''