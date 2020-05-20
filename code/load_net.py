# -*- coding: utf-8 -*-
import torch
from lstm_04 import LSTM

net = LSTM(6,10)
# print(net)
print(net.state_dict())
print('==============================load====================================')
net = torch.load('net_demo.pkl')
# print(net)
print(net.state_dict())