# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
from torch import nn
from  torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime


starttime = datetime.datetime.now()

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)

inFile = '/home/chen/Pycharm Projects/ITS/SCD_System_2.0/data/true_data/userTrueTrips_oneDay.csv'
userTrueTrips_df = pd.read_csv(inFile)
print(userTrueTrips_df)

userTrueTrips_df2 = userTrueTrips_df[userTrueTrips_df['outStation'] == 111]
print(userTrueTrips_df2)

trueOutflow_ls = [0 for x in range(38)]
for i in range(len(userTrueTrips_df2)):
    trueOutflow_ls[userTrueTrips_df2.iloc[i,3] - 11] += 1
print(trueOutflow_ls)

x = torch.unsqueeze(torch.linspace(11, 48, 38), dim=1)
y = torch.unsqueeze(torch.FloatTensor(trueOutflow_ls), dim=1)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


x_ = [0 for x in range(38)]
for i in range(1,len(trueOutflow_ls)):
    x_[i] = trueOutflow_ls[i-1]
x_ = torch.unsqueeze(torch.FloatTensor(x_), dim=1)

x_, y = Variable(x_), Variable(y)
# print(x)
print(x_)
print(y)

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
# print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=.2)
loss_func = nn.MSELoss()

for t in range(200):
    prediction = net(x_)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 50 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)






