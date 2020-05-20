# -*- coding: utf-8 -*-

import math


def number2radius(number):
    return number * math.pi / 180


def distance_between_twostations(inStation_loc, outStation_loc):
    lon1 = inStation_loc[0]
    lat1 = inStation_loc[1]
    lon2 = outStation_loc[0]
    lat2 = outStation_loc[1]
    deg_lat = number2radius(lat2 - lat1)
    deg_lon = number2radius(lon2 - lon1)
    a = math.pow(math.sin(deg_lat / 2), 2) + math.cos(number2radius(lat1)) * \
                                             math.cos(number2radius(lat2)) * math.pow(math.sin(deg_lon / 2), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371 * c

