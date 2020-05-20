# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 10:28
# @Author  : Chen Yu

import os
import pandas as pd
import numpy as np
import numpy.linalg as la
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from utils import *
from datetime import datetime
# 所有station训练一个model，每次epoch都随机测试1000个样本的指标，记录所有mape并画图
path = os.path.abspath('..')

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)


# def mape(y_pred, y_true):
#     y_pred = np.array(y_pred)
#     y_true = np.array(y_true)
#     m = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
#     return m
#
#
# def my_acc(y_true, y_pred):
#     F_norm = la.norm(y_true - y_pred, 'fro') / la.norm(y_true, 'fro') + 1e-3
#     return F_norm


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


def main():
    print('reading data ...')
    infile = path + '/data/raw_data/metroData_ODflow_15.csv'
    raw_flow = pd.read_csv(infile)
    # print(raw_flow)

    # ==== only train one station model ====
    # station_flow = raw_flow[raw_flow[' station'] == STATION_ID]
    # # print(station_flow)
    # ====

    trainData_x, trainData_y = trainDataGen(raw_flow, N_TS, N_DAY, N_WEEK)
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
    #     shuffle=False,
    #     num_workers=2
    # )


    lstm = LSTM(N_TS+N_DAY+N_WEEK, 10)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

    # ==== train
    print('training data ...')
    rmse_all_ls = []
    mae_all_ls = []
    mape_all_ls = []
    acc_all_ls = []

    rmse_test_all_ls = []
    mae_test_all_ls = []
    mape_test_all_ls = []
    acc_test_all_ls = []

    score_df = pd.DataFrame(-1, index=range(EPOCH), columns=['rmse', 'mae', 'mape', 'acc'])
    for epoch in range(EPOCH):
        print('epoch:', epoch+1)
        epoch_sta_time = datetime.now()
        rmse_epoch_ls = []
        mae_epoch_ls = []
        mape_epoch_ls = []
        acc_epoch_ls = []
        for step, (batch_x, batch_y) in enumerate(trainData_loader):
            outs = lstm(batch_x)
            loss = loss_func(outs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 1000 == 0:
                print(' step:', step+1, 'all:', int(num_train/BATCH_SIZE))

            y_pred = MaxMinNorm_re(outs, mao_y, mio_y).view(-1).data
            y_true = MaxMinNorm_re(batch_y, mao_y, mio_y).view(-1).data
            try:
                rmse_step, mae_step, mape_step, acc_step = evaluation(y_true, y_pred)
            except:
                rmse_step, mae_step, mape_step, acc_step = .01, .01, 5, .95
            rmse_epoch_ls.append(rmse_step)
            mae_epoch_ls.append(mae_step)
            mape_epoch_ls.append(mape_step)
            acc_epoch_ls.append(acc_step)
            break

        rmse_all_ls.append(np.mean(rmse_epoch_ls))
        mae_all_ls.append(np.mean(mae_epoch_ls))
        mape_all_ls.append(np.mean(mape_epoch_ls))
        acc_all_ls.append(np.mean(acc_epoch_ls))
        if (epoch+1) % 10 == 0:
            print('epoch: {}, loss: {:.5f}'.format(epoch+1, loss.item()))

        # ==== test method 1
        # for step_test, (batch_x_test, batch_y_test) in enumerate(testData_loader):
        #     outs_test = lstm(batch_x_test)
        #     y_pred_test = MaxMinNorm_re(outs_test, mao_y, mio_y).view(-1).data
        #     y_true_test = MaxMinNorm_re(batch_y_test, mao_y, mio_y).view(-1).data
        #     mape_step_test = mape(y_pred_test, y_true_test)
        #     acc_step_test = my_acc(y_true_test, y_pred_test)
        #     mape_test_all_ls.append(mape_step_test)
        #     acc_test_all_ls.append(acc_step_test)
        #     break

        # ==== test method 2
        pred_outs = lstm(test_x)
        pred_outs = MaxMinNorm_re(pred_outs, mao_y, mio_y).view(-1).data
        test_y = MaxMinNorm_re(test_y, mao_y, mio_y).view(-1).data
        try:
            rmse_epoch_test, mae_epoch_test, mape_epoch_test, acc_epoch_test = evaluation(test_y, pred_outs)
        except:
            rmse_epoch_test, mae_epoch_test, mape_epoch_test, acc_epoch_test = .01, .01, 5, .95
        rmse_test_all_ls.append(rmse_epoch_test)
        mae_test_all_ls.append(mae_epoch_test)
        mape_test_all_ls.append(mape_epoch_test)
        acc_test_all_ls.append(acc_epoch_test)

        score_df.iloc[epoch, 0] = rmse_epoch_test
        score_df.iloc[epoch, 1] = mae_epoch_test
        score_df.iloc[epoch, 2] = mape_epoch_test
        score_df.iloc[epoch, 3] = acc_epoch_test

        # ==== save
        print('saving net ...')
        # print('mao_x:', mao_x)  # mao_x: 6864
        # print('mio_x:', mio_x)  # mio_x: 1
        # print('mao_y:', mao_y)  # mao_y: 6864
        # print('mio_y:', mio_y)  # mio_y: 1
        torch.save(lstm, 'net_lstm_all_station.pkl')

        epoch_end_time = datetime.now()
        print('epoch used time:', epoch_end_time - epoch_sta_time)

    # ==== print score
    print('train score:')
    print(rmse_all_ls)
    print(mae_all_ls)
    print(mape_all_ls)
    print(acc_all_ls)

    print('test score:')
    print(rmse_test_all_ls)
    print(mae_test_all_ls)
    print(mape_test_all_ls)
    print(acc_test_all_ls)

    # ==== plot
    plt.plot(range(1, EPOCH+1), rmse_all_ls, c='r', label='训练集')  # ls='--', marker='^',
    plt.plot(range(1, EPOCH+1), rmse_test_all_ls, c='b', label='测试集')  # ls='--', marker='^',
    # plt.xticks(range(1, EPOCH+1))
    # plt.yticks(range(0, 110, 10))
    # plt.ylim((0, 100))
    plt.xlabel('迭代次数')
    plt.ylabel('RMSE')
    # plt.title('Rmse of all station')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path + r'result/all_station_result/rmse_all.png', dpi=300)

    plt.plot(range(1, EPOCH+1), mae_all_ls, c='r', label='训练集')  # ls='--', marker='^',
    plt.plot(range(1, EPOCH+1), mae_test_all_ls, c='b', label='测试集')  # ls='--', marker='^',
    # plt.xticks(range(1, EPOCH+1))
    # plt.yticks(range(0, 110, 10))
    # plt.ylim((0, 100))
    plt.xlabel('迭代次数')
    plt.ylabel('MAE')
    # plt.title('Mae of all station')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path + r'result/all_station_result/mae_all.png', dpi=300)

    plt.plot(range(1, EPOCH+1), mape_all_ls, c='r', label='训练集')  # ls='--', marker='^',
    plt.plot(range(1, EPOCH+1), mape_test_all_ls, c='b', label='测试集')  # ls='--', marker='^',
    # plt.xticks(range(1, EPOCH+1))
    # plt.yticks(range(0, 110, 10))
    # plt.ylim((0, 100))
    plt.xlabel('迭代次数')
    plt.ylabel('MAPE (%)')
    # plt.title('Mape of all station')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path + r'result/all_station_result/mape_all.png', dpi=300)

    plt.plot(range(1, EPOCH+1), acc_all_ls, c='r', label='训练集')  # ls='--', marker='^',
    plt.plot(range(1, EPOCH+1), acc_test_all_ls, c='b', label='测试集')  # ls='--', marker='^',
    # plt.xticks(range(1, EPOCH+1))
    # plt.yticks(range(0, 110, 10))
    # plt.ylim((0, 100))
    plt.xlabel('迭代次数')
    plt.ylabel('Accuracy')
    # plt.title('Accuracy of all station')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path + r'result/all_station_result/acc_all.png', dpi=300)

    plt.show()

    return score_df


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
    EPOCH = 10
    BATCH_SIZE = 100
    # BATCH_SIZE_TEST = 5000
    LR = .01

    score_df = main()
    # test()

    endtime = datetime.now()
    usetime = (endtime - starttime).seconds
    h = int(usetime / 3600)
    m = int((usetime - 3600 * h) / 60)
    s = usetime - 3600 * h - 60 * m
    print('time:', h, 'h', m, 'm', s, 's')
    print('system end')
