# -*- coding: utf-8 -*-
from mpl_toolkits.basemap import Basemap
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon
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
infile = '/home/chen/Pycharm Projects/ITS/SCD_System_2.0/data/true_data/station_mape_single_training.csv'
sta_mape_df = pd.read_csv(infile, index_col=0)
# print(sta_mape_df)

infile2 = '/home/chen/Pycharm Projects/ITS/SCD_System_2.0/data/raw_data/transferStations.pkl'
transferStation_info = pickle.load(open(infile2, 'rb'))
# print(transferStation_info[0])
# print(transferStation_info[0].keys())
# print(transferStation_info[0].values())
# transferstation_df = pd.DataFrame(data=transferStation_info[0].values(), index=transferStation_info[0].keys(),
#                                   columns=['includeStation', 'name'])
# print(transferstation_df)


# ==== plot map
fig = plt.figure(figsize=(8,6))
# fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
m = Basemap(llcrnrlon=120.91,llcrnrlat=31.00,urcrnrlon=122.12,urcrnrlat=31.33,
            projection='lcc',lat_1=33,lat_2=45,lon_0=100,ax=ax1)

m.drawmapboundary(fill_color='w')
m.fillcontinents(color='w',lake_color='w')
# m.drawcoastlines()
# m.drawcountries(linewidth=1.5)

shp_info = m.readshapefile('/home/chen/Pycharm Projects/ITS/practice/gadm36_CHN_shp/gadm36_CHN_3',
                           'states',drawbounds=False)

for info, shp in zip(m.states_info, m.states):
    proid = info['NAME_1']
    if proid == 'Shanghai':
        poly = Polygon(shp, facecolor='w', edgecolor='k', lw=.2)
        ax1.add_patch(poly)

for i in range(len(sta_mape_df)):
    lon, lat = m(sta_mape_df.iloc[i, 1], sta_mape_df.iloc[i, 2])
    m.plot(lon, lat, marker='o', markersize=get_logsize(sta_mape_df.iloc[i, 3]), color='k')



plt.xlabel('Shanghai Metro Flow Mape Map',fontsize=12)
plt.tight_layout()
plt.show()
