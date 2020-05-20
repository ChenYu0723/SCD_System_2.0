# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import random
from random import choice
import pickle

starttime = datetime.now()

# pd.set_option('display.height',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)

def ts2time(ts,date_ls):
    date_idx = int(ts/64)
    date = date_ls[date_idx]
    hour = int((ts - 64*date_idx)/4) + 6
    week = get_weekday(date)
    return str(date)[6:] + ' ' + week

def get_weekday(date):
    dic={0:'MON',1:'TUE',2:'WED',3:'THU',4:'FRI',5:'SAT',6:'SUN'}
    n = datetime.strptime("%s" % date, "%Y%m%d").weekday()
    return dic[n]

def main():
    print('reading data ...')
    infile = r'G:\Program\Pycharm Projects\File of Python3\SCD_System_2.0\data\true_data\metroData_ODflow_15.csv'
    raw_flow = pd.read_csv(infile)
    # print(raw_flow)
    date_ls = sorted(list(set(list(raw_flow['date']))))
    # print(date_ls)
    station_flow = raw_flow[raw_flow[' station']==STATION_ID]
    selected_date = range(STARTDAY,ENDDAY+1)
    station_flow = station_flow[station_flow['date'].isin(selected_date)]
    # print(station_flow)
    # station_flow.to_csv(r'G:\Program\Pycharm Projects\File of Python3\SCD_System_2.0\data\true_data\station_flow_%(sta_id)s_%(start)s_%(end)s.csv'
    #                     %{'sta_id':STATION_ID, 'start':STARTDAY, 'end':ENDDAY},index=False)
    selected_date_real = sorted(list(set(list(station_flow['date']))))
    print('selected date:',STARTDAY,'-',ENDDAY)
    print('real date:',selected_date_real)
    print('# real date:',len(selected_date_real))
    ts_ls = station_flow[' timeslot']


    print('plot...')
    inFlow = station_flow[' inFlow']
    outFlow = station_flow[' outFlow']
    fig = plt.figure(figsize=(30,5)) # 10 15 30
    plt.plot(ts_ls, inFlow, lw=.5)
    # plt.plot(ts_ls, outFlow, lw=.5)
    date_plot_ls = range(min(ts_ls), max(ts_ls)+1, int((60/15)*HOUR_INTERVAL))
    date_str_ls = []
    for t in date_plot_ls:
        date_str_ls.append(ts2time(t,date_ls))
    plt.xticks(date_plot_ls, date_str_ls, rotation=90)
    plt.title('Station Flow of %(sta_id)s from %(start)s to %(end)s' %{'sta_id':STATION_ID, 'start':STARTDAY, 'end':ENDDAY})
    plt.legend(fontsize = 15)
    plt.tight_layout()
    plt.savefig(r'G:\Program\Pycharm Projects\File of Python3\SCD_System_2.0\result\station_flow_%(sta_id)s_%(start)s_%(end)s.png'
                %{'sta_id':STATION_ID, 'start':STARTDAY, 'end':ENDDAY}, dpi=150)
    plt.show()










if __name__ == '__main__':
    print('system start')

    # interval = 15
    STATION_ID = 2011 #2035
    STARTDAY = 20170701
    ENDDAY = 20170731
    HOUR_INTERVAL = 8

    main()

    print('system end')


