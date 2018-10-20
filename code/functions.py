# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:48:10 2018

@author: bramv
"""
import math



def string_to_list(s, separator=','):
    l = []
    index = s.find(separator)
    while index!=-1:
        l.append(s[:index].strip())
        s = s[index+len(separator):]
        s = s.strip()
        index = s.find(separator)
    l.append(s)
    return l

def get_datalines(data):
    lines = []
    index = data.find('\n')
    while index!=-1:
        lines.append(data[:index])
        data = data[index:].strip()
        index = data.find('\n')
    lines.append(data)
    if lines[-1]=='': lines.pop()
    
    return lines

def list_data(data, separator=','):
    lines = get_datalines(data)
    lines_list = [string_to_list(j, separator) for j in lines]
    return lines_list

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def RK4(f, y, dt, t = None):
    #f = f(y, t = None)
    if not t is None:
        k1 = f(y, t)
        k2 = f(y + k1/2., t + dt/2.)
        k3 = f(y + k2/2., t + dt/2.)
        k4 = f(y + k3, t + dt)
    else:
        k1 = f(y)
        k2 = f(y + k1/2.)
        k3 = f(y + k2/2.)
        k4 = f(y + k3)
    return dt/6. * (k1 + 2.*k2 + 2.*k3 + k4) # = dy