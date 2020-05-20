# -*- coding: utf-8 -*-
from mpl_toolkits.basemap import Basemap
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
import folium
import pandas as pd
import pickle
# from SCD_System.code.station_list import station_ls
import datetime

starttime = datetime.datetime.now()

# pd.set_option('display.height',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)


def get_logsize(x):
    if x == 0:
        l = 0
    elif x < 0:
        l = 0
    else:
        l = math.log(x,2)
    return l


# ==== read data
infile = '/home/chen/Pycharm Projects/ITS/SCD_System_2.0/data/true_data/station_transInfo.csv'
station_transInfo = pd.read_csv(infile, index_col=0)
# print(station_transInfo)

infile2 = '/home/chen/Pycharm Projects/ITS/SCD_System_2.0/data/raw_data/transferStations.pkl'
transferStation_info = pickle.load(open(infile2, 'rb'))
# print(transferStation_info[0])
# print(transferStation_info[0].keys())
# print(transferStation_info[0].values())
# transferstation_df = pd.DataFrame(data=transferStation_info[0].values(), index=transferStation_info[0].keys(),
#                                   columns=['includeStation', 'name'])
# print(transferstation_df)


# ==== plot map
m = folium.Map(location=[31.10, 121.51])

# n = 0
# lim_ls = list(range(0+10*n, 10+10*n))
for i in range(len(station_transInfo)):
    # if i in lim_ls:
    lon, lat = station_transInfo.iloc[i, 1], station_transInfo.iloc[i, 2]
    folium.CircleMarker(location=[lat, lon], radius=3, popup='%s' %station_transInfo.iloc[i, 0]+'%s' %station_transInfo.iloc[i, 3],
                        color='#4169E1', fill=True, fill_color='#EE1289', ).add_to(m)
'''
# ==== plot all mape
fig = plt.figure()
plt.plot(station_transInfo['mape'])
# plt.xticks(range(len(sta_mape_df)),sta_mape_df['stationID'])
plt.xlabel('# Station')
plt.ylabel('Mape(%)')
plt.title("All station's mape")
plt.tight_layout()
# plt.savefig(r'/home/chen/Pycharm Projects/ITS/SCD_System_2.0/result/mape_all_station.png', dpi=150)
plt.show()
'''


m.save('station_transInfo.html')

print('end')