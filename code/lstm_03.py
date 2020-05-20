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
starttime = datetime.now()

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)


def trainDataGen(raw_data, n_ts, n_day, n_week=0):
    data = []
    l = len(raw_data)
    for i in range(l-n_day*64):
        if n_day==0 and i < n_ts:
            continue
        indata_day = []
        for j in range(n_day):
            indata_day.append(raw_data[i+j*64])
        indata_ts = raw_data[i+n_day*64-n_ts:i+n_day*64]
        outdata = raw_data[i+n_day*64:i+n_day*64+1]
        data.append((indata_day+indata_ts, outdata))
    return data


def toV(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_size)),
                Variable(torch.zeros(1, 1, self.hidden_size)))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        out = self.out(lstm_out.view(len(seq), -1))
        return out


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
    trainData = trainDataGen(inFlow_ls, N_TS, N_DAYS)
    print(trainData)
    print(len(trainData))

    lstm = LSTM(1, 10)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

    # ==== train
    for epoch in range(EPOCH):
        print('EPOCH:', epoch)
        count = 0
        for seq, outs in trainData[:21*64]:
            count += 1
            seq = toV(seq)
            outs = toV(outs)

            optimizer.zero_grad()
            lstm.hidden = lstm.init_hidden()
            trainOut = lstm(seq)
            loss = loss_func(trainOut, outs)
            if count % 50 == 0:
                pass
                # print('Loss:',loss.data.numpy())
            loss.backward()
            optimizer.step()

    # ==== test
    predData = []
    Loss = []
    for seq, trueVal in trainData[21*64:]:
        seq = toV(seq)
        trueVal = toV(trueVal)
        loss = loss_func(lstm(seq), trueVal)
        # print('Loss:', loss.data.numpy())
        Loss.append(loss.data.numpy())
        predData.append(lstm(seq)[-1].data.numpy()[0])

    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(len(inFlow_ls)), inFlow_ls, label='true data')
    plt.plot(range((N_DAYS+21)*64, len(inFlow_ls)), predData, label='prediction data')
    # plt.xlim(1500,2000)
    plt.text(1500, 5000, 'Loss=%.2f' % np.mean(Loss), fontdict={'size': 10, 'color': 'red'})
    plt.legend()
    plt.tight_layout()
    # plt.savefig(path + r'/result/station_flow_prediction.png', dpi=150)

    plt.show()


if __name__ == '__main__':
    print('system start')

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
    # a = list(range(256))
    # a_max = max(a)
    # a_min = min(a)
    # a = list(map(lambda x: x/(a_max-a_min),a))
    # print(a)
    # print(len(a))
    # b = trainDataGen(a,3,3)
    # print(b)
    # print(len(b))
    # for seq, outs in b[:1]:
    #     seq = toV(seq)
    #     print(seq)
    #     trainOut = seq.view(len(seq), 1, -1)
    #     print(trainOut)
    print('system end')
