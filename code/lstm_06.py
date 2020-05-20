# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datetime import datetime

path = os.path.abspath('..')

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)


def trainDataGen_demo(raw_data, n_ts, n_day, n_week=0):
    data_x = []
    data_y = []
    l = len(raw_data)
    for i in range(l-n_day*64):
        if n_day==0 and i < n_ts:
            continue
        indata_day = []
        for j in range(n_day):
            indata_day.append(raw_data[i+j*64])
        indata_ts = raw_data[i+n_day*64-n_ts:i+n_day*64]
        outdata = raw_data[i+n_day*64:i+n_day*64+1]
        # data.append((indata_day+indata_ts, outdata))
        data_x.append(indata_day+indata_ts)
        data_y.append(outdata)
    return data_x, data_y


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
                data_x.append(xxxx)

                yyyy.append(yyy[sta])
                data_y.append(yyyy)
        if (i+1)%10 == 0:
            print('dealed:',i+1,'all:',len(date_ls_train))
        if i+1 == 1:
            break
    return data_x, data_y


def MaxMinNorm(data):
    Max = np.max(data)
    Min = np.min(data)
    data = (data - Min)/(Max - Min)
    return data


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
    infile = path + '/data/true_data/metroData_ODflow_15.csv'
    raw_flow = pd.read_csv(infile)
    # print(raw_flow)
    trainData_x, trainData_y = trainDataGen(raw_flow, N_TS, N_DAY, N_WEEK)
    trainData_x = MaxMinNorm(trainData_x)
    trainData_y = MaxMinNorm(trainData_y)

    trainData_x = toV(trainData_x).view(len(trainData_x),1,N_TS+N_DAY+N_WEEK)
    trainData_y = toV(trainData_y).view(len(trainData_y),1,1)
    # print(trainData_x, len(trainData_x))
    # print(trainData_y, len(trainData_y))
    train_x = trainData_x[:1*64*322]
    train_y = trainData_y[:1*64*322]
    test_x = trainData_x[1*64*322:2*64*322]
    test_y = trainData_y[1*64*322:2*64*322]
    # print(train_x, len(train_x))
    # print(train_y, len(train_y))


    lstm = LSTM(N_TS+N_DAY+N_WEEK, 10)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

    # ==== train
    print('training data ...')
    for epoch in range(EPOCH):
        print('Epoch:', epoch+1)

        outs = lstm(train_x)
        # print(outs)
        # print(type(outs))
        loss = loss_func(outs, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1)%10 == 0:
            print('Epoch: {}, Loss: {:.5f}'.format(epoch+1, loss.item()))


    # ==== test
    print('testing data ...')
    lstm = lstm.eval()
    pred_outs = lstm(test_x)
    loss = loss_func(pred_outs, test_y)
    print('Test loss: {:.5f}'.format(loss.item()))

    test_y = test_y.view(-1).data.numpy()
    pred_outs = pred_outs.view(-1).data.numpy()

    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(len(test_x)), test_y, label='true data')
    plt.plot(range(len(test_x)), pred_outs, label='prediction data')
    plt.text(0, 0, 'Loss=%.5f' % loss.item(), fontdict={'size': 10, 'color': 'red'})
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig(path + r'/result/station_flow_prediction.png', dpi=150)

    plt.show()


def test():
    infile = path + '/data/true_data/metroData_ODflow_15.csv'
    raw_flow = pd.read_csv(infile)
    # print(raw_flow)
    trainData_x, trainData_y = trainDataGen(raw_flow, N_TS, N_DAY, N_WEEK)
    trainData_x = MaxMinNorm(trainData_x)
    trainData_y = MaxMinNorm(trainData_y)

    trainData_x = toV(trainData_x).view(len(trainData_x),1,N_TS+N_DAY+N_WEEK)
    trainData_y = toV(trainData_y).view(len(trainData_y),1,1)
    print(trainData_x, len(trainData_x))
    print(trainData_y, len(trainData_y))

    trainData_set = Data.TensorDataset(trainData_x, trainData_y)
    trainData_loader = Data.DataLoader(
        dataset=trainData_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    for epoch in range(1):
        for step, (batch_x, batch_y) in enumerate(trainData_loader):
            print('Epoch: ', epoch, '| Step: ', step)
            print('batch x: ')
            print(batch_x)
            print('batch y: ')
            print(batch_y)





if __name__ == '__main__':
    print('system start')
    starttime = datetime.now()

    # interval = 15
    # STATION_ID = 2011  # 2035 2011
    # STARTDAY = 20170701
    # ENDDAY = 20170731
    # HOUR_INTERVAL = 16

    N_TS = 3
    N_DAY = 3
    N_WEEK = 3
    EPOCH = 10
    BATCH_SIZE = 10
    LR = .01

    # main()
    test()

    endtime = datetime.now()
    usetime = (endtime - starttime).seconds
    h = int(usetime / 3600)
    m = int((usetime - 3600 * h) / 60)
    s = usetime - 3600 * h - 60 * m
    print('time:', h, 'h', m, 'm', s, 's')
    print('system end')
