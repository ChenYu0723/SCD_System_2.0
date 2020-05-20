# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from  torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime

path = os.path.abspath('..')
starttime = datetime.datetime.now()

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)


K = 2
EPOCH = 50
LR = .01
STATION_ID = 111

# ====
inFile = path + '/data/true_data/userTrueTrips_oneWK.csv'
userTrueTrips_df = pd.read_csv(inFile)
# print(userTrueTrips_df)

userTrueTrips_df2 = userTrueTrips_df[userTrueTrips_df['outStation'] == STATION_ID]
# print(userTrueTrips_df2)

date = sorted(list(set(list(userTrueTrips_df2['transDate']))))
date = date[:-2]
# print(date)

trueOutflow_ls = []
for day in date:
    userTrueTrips_df3 = userTrueTrips_df2[userTrueTrips_df2['transDate'] == day]
    # print(userTrueTrips_df3)
    ls = [0 for x in range(38)]
    for i in range(len(userTrueTrips_df3)):
        ls[userTrueTrips_df3.iloc[i,3] - 11] += 1
    trueOutflow_ls += ls
# print(trueOutflow_ls)
# print(len(trueOutflow_ls))

# plt.plot(trueOutflow_ls)
# plt.show()
# ====

def trainDataGen(seq,k):
    data = []
    l = len(seq)
    for i in range(l-k-1):
        indata = seq[i:i+k]
        outdata = seq[i+2:i+3]
        data.append((indata, outdata))
    return data

def toV(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

trainData = trainDataGen(trueOutflow_ls, K)
print(trainData)
# print(toV(trainData))
# ====

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        # nn.Dropout(.5)
        self.out = nn.Linear(hidden_size, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_size)),
                Variable(torch.zeros(1, 1, self.hidden_size)))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        out = self.out(lstm_out.view(len(seq), -1))
        return out

lstm = LSTM(1,10)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

for epoch in range(EPOCH):
    print('EPOCH:',epoch)
    count = 0
    for seq, outs in trainData[:152]:
        count +=1
        seq = toV(seq)
        outs = toV(outs)

        optimizer.zero_grad()
        lstm.hidden = lstm.init_hidden()
        trainOut = lstm(seq)
        loss = loss_func(trainOut,outs)
        if count % 50 == 0:
            pass
            # print('Loss:',loss.data.numpy())
        loss.backward()
        optimizer.step()

# ====
predData = []
Loss = []
for seq, trueVal in trainData[152:]:
    seq = toV(seq)
    trueVal = toV(trueVal)
    loss = loss_func(lstm(seq), trueVal)
    print('Loss:', loss.data.numpy())
    Loss.append(loss.data.numpy())
    predData.append(lstm(seq)[-1].data.numpy()[0])

fig = plt.figure()
plt.plot(range(190),trueOutflow_ls)
plt.plot(range(153,len(trueOutflow_ls)-K), predData)
plt.text(0.5, 0, 'Loss=%.4f' % np.mean(Loss), fontdict={'size':20, 'color':'red'})
plt.savefig(path + r'/result/station_flow_prediction.png', dpi=150)

plt.show()
