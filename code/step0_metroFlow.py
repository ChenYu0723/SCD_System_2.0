# -*- coding: utf-8 -*-
__author__ = 'xu'

import os, sys
import csv, pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random
import geojson
import copy

dataPath = "/media/xu/TOSHIBA EXT/SmartCard/data/"
dataPathOS = "/media/xu/TOSHIBA\ EXT/SmartCard/data/"


def sortMetroData():
    inFile = dataPathOS + 'SH_metro.csv'
    outFile = dataPathOS + 'SH_metro_sorted.csv'
    os.system('sort -t, -k1,1n -k5,5n -k6,6n ' + inFile + ' > ' + outFile)



# sort the raw metro data by time for demand inference
def sortTripsByTime():
    inFile = "/media/xu/OS/Data/SmartCardData/metroData.csv"
    outFile = "/media/xu/OS/Data/SmartCardData/metroData_ts.csv"
    os.system('sort -t, -k2,2n -k3,3n ' + inFile + ' > ' + outFile)



# sort metro data by user and time
def sortSelectedMetroData():
    inFile = "/media/xu/OS/Data/SmartCardData/metroData_selected.csv"
    outFile = 'results/metroData_sorted.csv'
    os.system('sort -t, -k1,1n -k3,3n -k4,4n ' + inFile + ' > ' + outFile)


# find the transfer metro stations
def transferStations():
    # load geojson map
    geojsonFile = open("StationsMap/metroStations.geojson", 'r')
    geoData = geojson.load(geojsonFile)

    stationInf = {}
    stationNames = {}
    for t in geoData['features']:
        try:
            stationID = int(t['properties']['ID'])
        except:
            continue
        lon, lat = t['geometry']['coordinates']
        stationName = t['properties']['NAME']
        stationInf[stationID] = (lon, lat)
        stationNames[stationID] = stationName

    # station ids having the same station name
    nameToID = {}
    for station in stationNames:
        name = stationNames[station]
        try:
            nameToID[name] += [station]
        except:
            nameToID[name] = [station]
    # assign new station id for the transfer stations 20xx
    transferStation = {}
    transferStationToNewID = {}
    count = 2000
    for name in nameToID:
        stations = nameToID[name]
        if len(stations) == 1:
            continue
        count += 1
        transferStation[count] = [stations, name]
        for station in stations:
            transferStationToNewID[station] = count
        print count, name, stations

    # save
    pickle.dump([transferStation, transferStationToNewID], open("results/transferStations.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)

    # update stationInf
    stationLoc = {}
    for station in stationInf:
        lon, lat = stationInf[station]
        if station in transferStationToNewID:
            station = transferStationToNewID[station]
        if station in stationLoc:
            continue
        else:
            stationLoc[station] = (lon, lat)

    # save to csv
    outData = open("results/metroStations.csv", 'wb')
    outData.writelines("stationID,lon,lat,name\n")
    for station in stationLoc:
        lon, lat = stationLoc[station]
        try:
            name = stationNames[station]
        except:
            name = transferStation[station][1]
            print name
        name = name.encode('utf8')
        stationID = str(station)
        lon = str(lon)
        lat = str(lat)
        outData.writelines(','.join([stationID, lon, lat, name]) + '\n')
    outData.close()


# travel demand matrix from metro data on 20170515 Monday
def travelDemand(interval):
    # interval = 10  # travel demand per 10 mins
    startHour = 6
    endHour = 22
    timeslotsPerDay = (endHour - startHour)*60/interval  # metro schedule

    dayIndex = {}
    count = 0
    for month in range(5, 9):
        yearmonth = '2017' + str(month).zfill(2)
        for day in range(32):
            sourceFile = "/media/xu/OS/Data/SmartCardData/SPTCC-JJYJY-" + yearmonth + str(day).zfill(2) + ".csv"
            if os.path.isfile(sourceFile):
                dayIndex[yearmonth+str(day).zfill(2)] = count
                count += 1
    # print dayIndex

    # all stations
    stationsData = open("results/metroStations.csv", 'r')
    stationsData.next()
    stations = []
    zeroFlow = {}
    for row in stationsData:
        row = row.rstrip().split(',')
        station = int(row[0])
        stations.append(station)
        zeroFlow[station] = 0

    transferStation, transferStationToNewID = pickle.load(open("results/transferStations.pkl", 'rb'))

    # sort stations
    stations.sort()

    
    # load data
    inFile = "/media/xu/OS/Data/SmartCardData/metroData_ts.csv"
    inData = open(inFile, 'r')
    inStationFlow = copy.deepcopy(zeroFlow)
    outStationFlow = copy.deepcopy(zeroFlow)
    currentTimeslot = 0
    currentDay = ''

    outFile = "/media/xu/OS/Data/SmartCardData/metroData_ODflow_" + str(interval) + ".csv"
    outData = open(outFile, 'wb')
    outData.writelines("date, timeslot, station, inFlow, outFlow\n")

    for row in inData:
        row = row.rstrip().split(',')
        userID = int(row[0])
        transDay = row[1]
        transTime = row[2]
        transHour = int(transTime[:2])
        if transHour < startHour or transHour >= endHour:
            continue
        minute = int(transTime[2:4])
        try:
            dayIdx = dayIndex[transDay]
        except:
            continue

        timeslot = dayIdx * timeslotsPerDay + (transHour - startHour) * 60 / interval + minute / interval
        if timeslot < 0:
            continue

        if currentDay == '':
            currentDay = transDay

        if timeslot != currentTimeslot:
            # save the od matrix of the last timeslot
            for station in stations:
                # print currentDay, currentTimeslot
                inFlow = str(inStationFlow[station])
                outFlow = str(outStationFlow[station])
                outData.writelines(','.join([currentDay, str(currentTimeslot), str(station), inFlow, outFlow]) + '\n')
            # initilize
            inStationFlow = copy.deepcopy(zeroFlow)
            outStationFlow = copy.deepcopy(zeroFlow)
            currentTimeslot = timeslot
            currentDay = transDay

        inStation = int(row[3])
        outStation = int(row[4])

        try:
            inStation_new = transferStationToNewID[inStation]
        except:
            inStation_new = inStation
        try:
            outStation_new = transferStationToNewID[outStation]
        except:
            outStation_new = outStation

        if inStation_new not in set(stations) or outStation_new not in set(stations):
            continue

        inStationFlow[inStation_new] += 1
        outStationFlow[outStation_new] += 1

    # last timeslot
    # save the od matrix of the last timeslot
    for station in stations:
        inFlow = str(inStationFlow[station])
        outFlow = str(outStationFlow[station])
        outData.writelines(','.join([transDay, str(timeslot), str(station), inFlow, outFlow]) + '\n')

    inData.close()
    outData.close()


def metroStationMap():
    # load geojson map
    geojsonFile = open("../StationsMap/metroStations.geojson", 'r')
    geoData = geojson.load(geojsonFile)

    stations = []
    for t in geoData['features']:
        try:
            stationID = int(t['properties']['ID'])
        except:
            continue
        lon, lat = t['geometry']['coordinates']
        stations.append([stationID, lon, lat])

    # save to csv
    outData = open("results/metroStations.csv", 'wb')
    outData.writelines("stationID,lon,lat\n")
    for row in stations:
        stationID, lon, lat = row
        stationID = str(stationID)
        lon = str(lon)
        lat = str(lat)
        outData.writelines(','.join([stationID, lon, lat]) + '\n')
    outData.close()




def main():
    yearmonth = '201507'

    travelDemand(15)





if __name__ == '__main__':
    main()
