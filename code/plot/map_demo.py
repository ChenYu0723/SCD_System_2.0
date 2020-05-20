# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
bmap = Basemap(llcrnrlon=120.71,llcrnrlat=30.80,urcrnrlon=122.12,urcrnrlat=31.43,
               projection='lcc', lat_1=33, lat_2=45, lon_0=100, ax=ax1)
shp_info = bmap.readshapefile('G:\Program\Pycharm Projects\File of Python3\Geo_info\gadm36_CHN_shp\gadm36_CHN_3',
                              'states', drawbounds=False)

for info, shp in zip(bmap.states_info, bmap.states):
    proid = info['NAME_1']
    if proid == 'Shanghai':
        poly = Polygon(shp, facecolor='w', edgecolor='b', lw=0.2)
        ax1.add_patch(poly)

bmap.drawcoastlines()
bmap.drawcountries()
bmap.drawparallels(np.arange(23, 29, 2), labels=[1, 0, 0, 0])
bmap.drawmeridians(np.arange(115, 121, 2), labels=[0, 0, 0, 1])
plt.title('Fujian Province')
# plt.savefig('fig_province.png', dpi=100, bbox_inches='tight')
plt.show()
plt.clf()
plt.close()