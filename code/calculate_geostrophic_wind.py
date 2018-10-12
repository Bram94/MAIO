# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:26:09 2018

@author: bramv
"""
import requests
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import settings as s
import functions as ft



class GeoWind():
    def __init__(self):
        pass
    
    

def calculate_geostrophic_wind():
    gw_data = GeoWind()
    
    filename = s.data_path+'KNMI_'+s.year+s.month+'_hourly_pressure.txt'
    
    if not os.path.exists(filename):
        url = 'http://projects.knmi.nl/klimatologie/uurgegevens/getdata_uur.cgi'
        r = requests.post(url, data={"stns":'ALL', "start":s.year+s.month+"0101", "end":s.year+s.month+str(s.n_days)+"24", "vars":"P"})
        content = r.text
        with open(filename, 'w') as f:
            f.write(content)
    
    
    
    data = np.char.strip(np.loadtxt(filename, dtype='str', delimiter = ','))
    
    stations = data[:,0].astype('int')
    #The stations might not be sorted, and np.unique returns a sorted array by default, which is
    #not desired here. With the method below this is circumvented.
    stations_unique_indices = np.unique(stations, return_index = True)[1]
    stations_unique = stations[np.sort(stations_unique_indices)] 
    dates = data[:,1]
    hours = data[:,2].astype('int')
    pressures = data[:,3]
    pressures[pressures == ''] = np.nan
    pressures = pressures.astype('float') / 10.
    
    #Reshape the pressure array, in such a way that observations for different stations and dates are grouped.
    index = np.where(np.abs(stations[1:] - stations[:-1]) > 0)[0][0] + 1
    n_times = 24
    n_dates = int(index / 24)
    n_stations = int(len(data) / (n_dates * n_times))
    pressures = np.reshape(pressures, (n_stations, n_dates, n_times))
    pressures = {stations_unique[j] : pressures[j] for j in range(n_stations)}
    #print(list(pressures.keys()))
    
    #Read coordinates of the stations
    with open(filename,'r') as f:
        text = f.read()
        i1 = text.index('NAME')+6
        i2 = text.index('YYYYMMDD')-11
        
        coords_data = ft.list_data(text[i1:i2+5], separator = ' ')
        coords = {int(j[1][:-1]) : np.array([j[2], j[3]], dtype = 'float') for j in coords_data}
    
    cabauw_id = int([j[1][:-1] for j in coords_data if j[-1] == 'CABAUW'][0])
    
    distances_to_cabauw = {j: ft.haversine(coords[cabauw_id][0], coords[cabauw_id][1], coords[j][0], coords[j][1]) for j in coords if not j == cabauw_id}
    max_distance = 75 #km
    selected_stations = [j for j in distances_to_cabauw if distances_to_cabauw[j] < max_distance and j in pressures and not np.isnan(pressures[j][0][0])]
    print('selected_stations: ',[j[-1] for j in coords_data if int(j[1][:-1]) in selected_stations and distances_to_cabauw[int(j[1][:-1])] < 75])
    
    
    selected_pressures = np.zeros((n_dates, n_times, len(selected_stations)))
    for j in range(len(selected_stations)):
        selected_pressures[:, :, j] = pressures[selected_stations[j]]
        
    selected_coords = np.array([coords[j] for j in selected_stations])[:,::-1] * np.pi/180. #Swap latitude and longitude
        
    def model(coords, dpdphi, dpdlambda):
        return p_0 + dpdphi * (coords[:,0] - phi_0) + dpdlambda * (coords[:,1] - lambda_0)
    
    rho = 1.25   
    f = 2*7.29e-5 * np.sin(coords[cabauw_id][1] * np.pi/180.)
    R = 6371e3
    p_gradient = np.zeros((n_dates, n_times, 2))
    for i in range(0, n_dates):
        for j in range(0, n_times):
            p_0 = pressures[cabauw_id][i][j]
            phi_0 = coords[cabauw_id][1] * np.pi/180.
            lambda_0 = coords[cabauw_id][0] * np.pi/180.
            estimates = curve_fit(model, selected_coords, selected_pressures[i, j], [0,0])
            p_gradient[i, j] = estimates[0]
            
    #*100 is to convert pressures from hPa to Pa
    V_g = 1. / (rho * f * R) * 100. * p_gradient * np.array([[[-1., 1./np.cos(coords[cabauw_id][1] * np.pi/180.)]]])
    """V_g is now available every hour, from hour 1 to 24, while observations at Cabauw are available every 10 minutes.
    V_g therefore needs to be interpolated to the times at which Cabauw observations are available.
    This is done by taking for each Cabauw observation the pressure at the nearest hour.
    """
    V_g_interpolated = np.zeros((V_g.shape[0], V_g.shape[1] * 6, 2))
    V_g_interpolated[:, :3] = V_g[:,np.newaxis, 0]; V_g_interpolated[:, -3:] = V_g[:,np.newaxis, -1]
    
    for i in range(0, V_g.shape[1] - 1):
        V_g_interpolated[:, 3 + 6*i : 9 + 6*i] = V_g[:, np.newaxis, i]

    gw_data.V_g = V_g
    gw_data.V_g_interpolated = V_g_interpolated
    gw_data.dates = dates
    gw_data.hours = hours
    
    return gw_data


if __name__ == '__main__':
    #This part is not executed when importing this script, only when running this as the main script.
    fig, ax = plt.subplots(1, 2)
    
    gw_data = calculate_geostrophic_wind()
    V_g_speed = np.linalg.norm(gw_data.V_g, axis = 2)
    V_g_speed_interpolated = np.linalg.norm(gw_data.V_g_interpolated, axis = 2)

    ax[0].plot(gw_data.dates[:744].astype(float) + gw_data.hours[:744].astype(float) / 24, V_g_speed.flatten())
    ax[1].plot(V_g_speed_interpolated.flatten())
    plt.show()