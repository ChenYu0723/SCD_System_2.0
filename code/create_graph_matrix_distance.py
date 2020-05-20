# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import geojson
from calculate_distance import *

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',500)

path = os.path.abspath('..')
'''
sta_linecolor = {'浦江线':'#aeaeae', '轨道交通10号线':'#c7afd3', '轨道交通11号线':'#8e2323', '轨道交通12号线':'#007b63', '轨道交通13号线':'#f293d1', '轨道交通16号线':'#32d4cb', '轨道交通17号线':'#be7970', '轨道交通1号线':'#ee3229', '轨道交通2号线':'#36b854', '轨道交通3号线':'#ffd823', '轨道交通4号线':'#320177', '轨道交通5号线':'#823094', '轨道交通6号线':'#ce047a', '轨道交通7号线':'#f3560f', '轨道交通8号线':'#008cc1', '轨道交通9号线':'#91c5db'}

infile = path + '/data/raw_data/stations/shanghai_metro.geojson'
geoData = geojson.load(open(infile, 'r'))
# data = geoData['features'][0]['properties']
# print(data)
# print(len(data))
data = geoData
data_dic = dict(data['features'][0]['properties'],**data['features'][0]['geometry'])
# print(data_dic)
col = data_dic.keys()
stationInfo_all_df = pd.DataFrame(columns=col)
for i in range(len(data['features'])):
    dic = dict(data['features'][i]['properties'],**data['features'][i]['geometry'])
    stationInfo_all_df = stationInfo_all_df.append(dic, ignore_index=True)
# print(stationInfo_all_df)
col_sle = ['NAME', 'LINEBELONG', 'ID', 'CHANGETYPE']
stationInfo_simple_df = pd.DataFrame(stationInfo_all_df, columns=col_sle)
for i in range(len(stationInfo_simple_df)):
    if stationInfo_simple_df.iloc[i,2] == None:
        stationInfo_simple_df.iloc[i, 2] = '-1'
stationInfo_simple_df['ID'] = list(map(lambda x:int(x), stationInfo_simple_df['ID']))
# print(stationInfo_simple_df)
# print(stationInfo_simple_df['ID'])

# stationInfo_singleLine_df = stationInfo_simple_df[stationInfo_simple_df['LINEBELONG'] == '轨道交通8号线']
# stationInfo_singleLine_df['ID'] = list(map(lambda x:int(x), stationInfo_singleLine_df['ID']))
# stationInfo_singleLine_df = stationInfo_singleLine_df.sort_values(by='ID', ascending=True)
# print(stationInfo_singleLine_df)


infile2 = path + '/data/raw_data/transferStations.pkl'
transferStation_info = pickle.load(open(infile2, 'rb'))
transferstation_df = pd.DataFrame(data=transferStation_info[0].values(), index=transferStation_info[0].keys(),
                                  columns=['includeStation', 'name'])
# print(transferstation_df)

def getNewID(data):
    newid = -1
    for i in range(len(ls)):
        if 243 in ls[i]:
            print(i)
            break

'''
infile3 = path + '/data/raw_data/metroStations.csv'
station_id_df = pd.read_csv(infile3)
# print(station_id_df)
'''
station_id_df['name'] = ''
station_id_df['includeStation'] = ''
station_id_df['neighborStation'] = ''
'''
'''
for i in range(len(station_id_df)):
    sta_id = station_id_df.iloc[i, 0]
    neighbor = []
    if sta_id > 2000:
        station_id_df.iloc[i, 3] = transferstation_df.loc[sta_id, 'name']
        include = transferstation_df.loc[sta_id, 'includeStation']
        station_id_df.iloc[i, 4] = str(include)
        for n in range(len(include)):
            a = include[n]-1
            b = include[n]+1
            if a in list(stationInfo_simple_df['ID']):
                neighbor.append(a)
            if b in list(stationInfo_simple_df['ID']):
                neighbor.append(b)
        station_id_df.iloc[i, 5] = str(neighbor)

    else:
        for j in range(len(stationInfo_simple_df)):
            try:
                if int(stationInfo_simple_df.iloc[j, 2]) == sta_id:
                    station_id_df.iloc[i, 3] = stationInfo_simple_df.iloc[j, 0]
            except:
                pass
        a = sta_id - 1
        b = sta_id + 1
        if a in list(stationInfo_simple_df['ID']):
            neighbor.append(a)
        if b in list(stationInfo_simple_df['ID']):
            neighbor.append(b)
        station_id_df.iloc[i, 5] = str(neighbor)

# print(station_id_df)

for i in range(len(station_id_df)):
    neighbor = eval(station_id_df.iloc[i, 5])
    for j in range(len(neighbor)):
        curid = neighbor[j]
        ls = list(transferstation_df['includeStation'])
        newid = 0
        for k in range(len(ls)):
            if curid in ls[k]:
                newid = transferstation_df.index[k]
        if newid:
            neighbor[j] = newid
    neighbor = list(set(neighbor))
    station_id_df.iloc[i, 5] = str(neighbor)

# print(station_id_df)

# search_name = '虹桥火车站'
# df_result = station_id_df[station_id_df['name'] == search_name]
# print(df_result)
'''
'''
correct_ind_ls = [55, 13, 53, 90]
correct_ID_ls = [1045, 1020, 1134, 1120]
for i in range(len(correct_ind_ls)):
    neighbor = eval(station_id_df.iloc[correct_ind_ls[i], 5])
    neighbor.append(correct_ID_ls[i])
    station_id_df.iloc[correct_ind_ls[i], 5] = str(neighbor)

# print(station_id_df)

# all_ls = []
# for i in range(len(station_id_df)):
#     ls = eval(station_id_df.iloc[i,-1])
#     for j in range(len(ls)):
#         all_ls.append(ls[j])
# all_ls = list(set(all_ls))
# print(len(all_ls))

# search_name = '嘉定新城'
# df_result = station_id_df[station_id_df['name'] == search_name]
# print(df_result)

# station_id_df.to_csv('/home/chen/Pycharm Projects/ITS/SCD_System_2.0/data/true_data/station_transInfo.csv')
'''
station_id_ls = sorted(list(station_id_df['stationID']))
# print(station_id_ls)
graph_matrix_df = pd.DataFrame(0, index=station_id_ls, columns=station_id_ls)
# print(graph_matrix_df)
for i in range(len(station_id_df)):
    sta_id_a = station_id_df.iloc[i, 0]
    sta_loc_a = []
    sta_loc_a.append(station_id_df.iloc[i, 1])
    sta_loc_a.append(station_id_df.iloc[i, 2])
    for j in range(len(station_id_df)):
        sta_id_b = station_id_df.iloc[j, 0]
        sta_loc_b = []
        sta_loc_b.append(station_id_df.iloc[j, 1])
        sta_loc_b.append(station_id_df.iloc[j, 2])
        dis = distance_between_twostations(sta_loc_a, sta_loc_b)
        graph_matrix_df.loc[sta_id_a, sta_id_b] = dis
    if i % 10 ==0:
        print('dealed:', i, 'all:', len(station_id_df))

    # neighbor = eval(station_id_df.iloc[i, 5])
    # for j in range(len(neighbor)):
    #     neighbor_id = neighbor[j]
    #     graph_matrix_df.loc[sta_id, neighbor_id] = 1

# print(graph_matrix_df)
graph_matrix_df.to_csv('/home/chen/Pycharm Projects/ITS/SCD_System_2.0/data/true_data/station_adj_dis.csv', header=False, index=False)

# graph_matrix = graph_matrix_df.to_numpy()
# print(graph_matrix)
# print((graph_matrix == graph_matrix.T).all())
