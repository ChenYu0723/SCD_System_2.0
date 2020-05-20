# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

path = os.path.abspath('..')

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)


def trainDataGen(raw_data, n_ts, n_day, n_week=0):
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
    date_ls = sorted(list(set(list(raw_flow['date']))))
    # print(date_ls)
    station_flow = raw_flow[raw_flow[' station'] == STATION_ID]
    selected_date = range(STARTDAY, ENDDAY+1)
    station_flow = station_flow[station_flow['date'].isin(selected_date)]
    # print(station_flow)
    selected_date_real = sorted(list(set(list(station_flow['date']))))
    # print('selected date:',STARTDAY,'-',ENDDAY)
    # print('real date:',selected_date_real)
    # print('# real date:',len(selected_date_real))
    ts_ls = station_flow[' timeslot']

    inFlow_ls = list(station_flow[' inFlow'])
    # print(inFlow_ls)
    inFlow_ls_max = max(inFlow_ls)
    inFlow_ls = list(map(lambda x: x/inFlow_ls_max,inFlow_ls))
    # print(inFlow_ls)

    trainData_x, trainData_y = trainDataGen(inFlow_ls, N_TS, N_DAYS)
    print(trainData_x)
    trainData_x = toV(trainData_x).view(len(trainData_x),1,6)
    trainData_y = toV(trainData_y).view(len(trainData_y),1,1)
    # print(trainData_x, len(trainData_x))
    # print(trainData_y, len(trainData_y))
    train_x = trainData_x[:21*64]
    train_y = trainData_y[:21*64]
    test_x = trainData_x[21*64:]
    test_y = trainData_y[21*64:]
    # print(train_x, len(train_x))
    # print(train_y, len(train_y))


    lstm = LSTM(6, 10)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

    # ==== train
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

    # # ==== save
    # print('saving net ...')
    # torch.save(lstm, 'net_demo.pkl')


    # ==== test
    lstm = lstm.eval()
    pred_outs = lstm(test_x)
    loss = loss_func(pred_outs, test_y)
    print('Test loss: {:.5f}'.format(loss.item()))

    pred_outs = pred_outs.view(-1).data.numpy()

    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(len(inFlow_ls)), inFlow_ls, label='true data')
    plt.plot(range((N_DAYS+21)*64, len(inFlow_ls)), pred_outs, label='prediction data')
    plt.text(1750, 0, 'Loss=%.5f' % loss.item(), fontdict={'size': 10, 'color': 'red'})
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig(path + r'/result/station_flow_prediction.png', dpi=150)

    plt.show()


def test():
    a = list(range(256))
    # a_max = max(a)
    # a_min = min(a)
    # a = list(map(lambda x: x/(a_max-a_min),a))
    # print(a)
    # print(len(a))
    x, y = trainDataGen(a,3,3)
    print(x)
    print(len(x))
    print(y)
    print(len(y))
    x = toV(x)
    y = toV(y)
    x = x.view(len(x),1,6)
    y = y.view(len(y),1,1)
    print(x)
    print(len(x))
    print(y)


if __name__ == '__main__':
    print('system start')
    starttime = datetime.now()

    # interval = 15
    STATION_ID = 2011  # 2035 2011
    STARTDAY = 20170701
    ENDDAY = 20170731
    HOUR_INTERVAL = 16

    N_TS = 3
    N_DAYS = 3
    EPOCH = 10
    LR = .01

    main()
    # test()

    endtime = datetime.now()
    print('time:', (endtime - starttime).seconds, 's')
    usetime = (endtime - starttime).seconds
    h = int(usetime / 3600)
    m = int((usetime - 3600 * h) / 60)
    s = usetime - 3600 * h - 60 * m
    print('time:', h, 'h', m, 'm', s, 's')
    print('system end')
